#!/usr/bin/env python3
"""
数据目录扫描和向量数据库重建模块
供Streamlit前端调用
"""

import os
import glob
import time
import shutil


def scan_data_directory(data_dir="./data/"):
    """扫描data目录，返回目录结构和文件统计信息（供Streamlit前端显示）"""
    
    result = {
        "success": True,
        "message": "",
        "start_time": time.time(),
        "directories": [],
        "total_files": 0,
        "file_breakdown": {
            "pdf": 0,
            "doc": 0,
            "docx": 0
        }
    }
    
    if not os.path.exists(data_dir):
        result["success"] = False
        result["message"] = f"数据目录不存在: {data_dir}"
        return result
    
    # 目录映射
    category_mapping = {
        "法律": "National Laws",
        "地方法律 法规规章": "Local Regulations",
        "行政法规": "Administrative Regulations",
        "规章及法律规范": "Rules and Norms",
    }
    
    extensions = ["*.pdf", "*.doc", "*.docx"]
    
    # 扫描data目录下的直接文件
    for ext in extensions:
        pattern = os.path.join(data_dir, ext)
        files = glob.glob(pattern)
        if files:
            ext_name = ext.replace("*.", "")
            result["file_breakdown"][ext_name] += len(files)
            result["total_files"] += len(files)
            result["directories"].append({
                "name": "根目录文件",
                "category": "General Documents",
                "files": files,
                "count": len(files),
                "file_types": {ext_name: len(files)}
            })
    
    # 遍历子目录
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        
        if not os.path.isdir(subdir_path):
            continue
        
        category = category_mapping.get(subdir, subdir)
        dir_info = {
            "name": subdir,
            "category": category,
            "files": [],
            "count": 0,
            "file_types": {"pdf": 0, "doc": 0, "docx": 0}
        }
        
        for ext in extensions:
            pattern = os.path.join(subdir_path, "**", ext)
            files = glob.glob(pattern, recursive=True)
            ext_name = ext.replace("*.", "")
            dir_info["files"].extend(files)
            dir_info["file_types"][ext_name] = len(files)
            dir_info["count"] += len(files)
            result["file_breakdown"][ext_name] += len(files)
            result["total_files"] += len(files)
        
        result["directories"].append(dir_info)
    
    result["end_time"] = time.time()
    result["elapsed"] = result["end_time"] - result["start_time"]
    
    return result


def load_single_file(file_path, category, timeout=60):
    """加载单个文件并返回文档列表（使用LibreOffice处理doc/docx），超时60秒自动跳过"""
    from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
    from langchain_core.documents import Document
    import threading
    import queue
    
    docs = []
    file_name = os.path.basename(file_path)
    result_queue = queue.Queue()
    
    def load_with_timeout():
        """在子线程中执行加载操作"""
        try:
            if file_path.endswith('.pdf'):
                loader = PyMuPDFLoader(file_path)
                file_type = "PDF"
            elif file_path.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(file_path)
                file_type = "DOCX"
            elif file_path.endswith('.doc'):
                loader = UnstructuredWordDocumentLoader(file_path)
                file_type = "DOC"
                print(f"[LibreOffice] 正在转换: {file_name}")
            else:
                result_queue.put(("skip", [], None))
                return
            
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["category"] = category
                doc.metadata["file_type"] = os.path.splitext(file_path)[1]
            result_queue.put(("success", loaded_docs, file_type))
        except UnicodeDecodeError as e:
            result_queue.put(("unicode_error", [], str(e)))
        except Exception as e:
            result_queue.put(("error", [], str(e)))
    
    # 启动加载线程
    load_thread = threading.Thread(target=load_with_timeout, daemon=True)
    load_thread.start()
    
    # 等待结果或超时
    load_thread.join(timeout=timeout)
    
    if load_thread.is_alive():
        # 超时，终止线程
        print(f"[超时跳过] {file_name}: 处理超过{timeout}秒")
        return docs
    
    # 获取结果
    try:
        status, loaded_docs, file_type = result_queue.get_nowait()
        
        if status == "success":
            docs.extend(loaded_docs)
            print(f"[成功] {file_type}: {file_name} ({len(loaded_docs)} 页)")
            return docs
        elif status == "unicode_error" or status == "error":
            err_msg = loaded_docs[0] if loaded_docs else "未知错误"
            print(f"[尝试备用方案] {file_name}: {err_msg}")
        else:
            return docs
    except queue.Empty:
        return docs
    
    # 备用方案：使用 docx2txt 或直接读取二进制
    if file_path.endswith('.doc'):
        # 方案1: 尝试用 docx2txt
        try:
            import docx2txt
            
            def load_docx2txt():
                try:
                    text = docx2txt.process(file_path)
                    result_queue.put(("docx2txt_success", text, None))
                except Exception as e:
                    result_queue.put(("docx2txt_error", str(e), None))
            
            t = threading.Thread(target=load_docx2txt, daemon=True)
            t.start()
            t.join(timeout=timeout)
            
            if t.is_alive():
                print(f"[超时跳过] {file_name}: docx2txt处理超过{timeout}秒")
            else:
                try:
                    status, text, _ = result_queue.get_nowait()
                    if status == "docx2txt_success" and text and len(text.strip()) > 10:
                        doc = Document(
                            page_content=text,
                            metadata={"category": category, "file_type": ".doc", "source": file_path}
                        )
                        docs.append(doc)
                        print(f"[备用方案] {file_name}: 使用docx2txt成功 ({len(text)} 字符)")
                        return docs
                except queue.Empty:
                    pass
        except:
            pass
        
        # 方案2: 使用 subprocess 调用 soffice 直接转换
        try:
            import subprocess
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            temp_input = os.path.join(temp_dir, "input.doc")
            shutil.copy(file_path, temp_input)
            
            soffice_cmd = "D:\\1 下载\\LibreOffice\\program\\soffice.exe"
            result = subprocess.run([
                soffice_cmd, "--headless", "--convert-to", "txt:Text",
                "--outdir", temp_dir, temp_input
            ], capture_output=True, timeout=timeout)
            
            output_file = os.path.join(temp_dir, "input.txt")
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                if text and len(text.strip()) > 10:
                    doc = Document(
                        page_content=text,
                        metadata={"category": category, "file_type": ".doc", "source": file_path}
                    )
                    docs.append(doc)
                    print(f"[备用方案] {file_name}: 使用soffice直接转换成功 ({len(text)} 字符)")
            
            shutil.rmtree(temp_dir, ignore_errors=True)
            return docs
        except subprocess.TimeoutExpired:
            print(f"[超时跳过] {file_name}: soffice转换超过{timeout}秒")
        except Exception as e2:
            print(f"[备用方案失败] {file_name}: {e2}")
    
    return docs


def rebuild_vectorstore_with_progress(data_dir="./data/", persist_dir="./chroma_db_pro/", 
                                        progress_callback=None):
    """重建向量数据库，支持进度回调（供Streamlit前端显示）"""
    from langchain_chroma import Chroma
    
    # 延迟导入，避免在模块加载时就初始化整个后端
    import sys
    import importlib
    
    # 从labor_law_complete_fixed获取embeddings和LegalRegexSplitter
    try:
        backend = importlib.import_module("labor_law_complete_fixed")
        embeddings = backend.embeddings
        LegalRegexSplitter = backend.LegalRegexSplitter
    except Exception as e:
        if progress_callback:
            progress_callback({
                "status": "error",
                "message": f"无法加载后端模块: {str(e)}",
                "progress": 0
            })
        return {"success": False, "message": f"无法加载后端模块: {str(e)}"}
    
    start_time = time.time()
    
    result = {
        "success": True,
        "message": "",
        "start_time": start_time,
        "documents_loaded": 0,
        "chunks_created": 0,
        "files_processed": 0,
        "errors": [],
        "directory_stats": []
    }
    
    if progress_callback:
        progress_callback({
            "status": "scanning",
            "message": "正在扫描数据目录...",
            "progress": 0
        })
    
    # 扫描目录
    scan_result = scan_data_directory(data_dir)
    if not scan_result["success"]:
        result["success"] = False
        result["message"] = scan_result["message"]
        return result
    
    result["directory_stats"] = scan_result["directories"]
    total_files = scan_result["total_files"]
    
    if progress_callback:
        progress_callback({
            "status": "scanning",
            "message": f"扫描完成，发现 {total_files} 个文件",
            "progress": 5,
            "details": scan_result
        })
    
    if total_files == 0:
        result["success"] = False
        result["message"] = "未发现任何文档文件"
        return result
    
    # 加载所有文档
    all_documents = []
    category_mapping = {
        "法律": "National Laws",
        "地方法律 法规规章": "Local Regulations",
        "行政法规": "Administrative Regulations",
        "规章及法律规范": "Rules and Norms",
    }
    
    processed_files = 0
    errors = []
    
    # 遍历所有目录和文件
    for dir_info in scan_result["directories"]:
        dir_name = dir_info["name"]
        category = dir_info["category"]
        files = dir_info["files"]
        
        if progress_callback:
            progress_callback({
                "status": "loading",
                "message": f"正在加载 {dir_name}...",
                "progress": 5 + int(45 * processed_files / max(total_files, 1)),
                "current_dir": dir_name,
                "files_done": processed_files,
                "files_total": total_files
            })
        
        for file_path in files:
            try:
                docs = load_single_file(file_path, category)
                all_documents.extend(docs)
                processed_files += 1
                
                # 每处理20个文件报告一次进度
                if processed_files % 20 == 0 and progress_callback:
                    progress_callback({
                        "status": "loading",
                        "message": f"已加载 {processed_files}/{total_files} 个文件",
                        "progress": 5 + int(45 * processed_files / max(total_files, 1)),
                        "files_done": processed_files,
                        "files_total": total_files
                    })
            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")
    
    result["documents_loaded"] = len(all_documents)
    result["files_processed"] = processed_files
    result["errors"] = errors[:50]  # 只保留前50个错误
    
    if progress_callback:
        progress_callback({
            "status": "splitting",
            "message": f"加载完成，共 {len(all_documents)} 个文档，正在切分...",
            "progress": 50
        })
    
    # 切分文档
    if all_documents:
        splitter = LegalRegexSplitter()
        splits = splitter.split_documents(all_documents)
        result["chunks_created"] = len(splits)
        
        if progress_callback:
            progress_callback({
                "status": "splitting",
                "message": f"切分完成，共 {len(splits)} 个切块，正在写入向量数据库...",
                "progress": 70
            })
        
        # 创建/清空向量数据库
        if os.path.exists(persist_dir):
            if progress_callback:
                progress_callback({
                    "status": "clearing",
                    "message": "正在清空旧数据库...",
                    "progress": 75
                })
            try:
                shutil.rmtree(persist_dir)
            except Exception as e:
                errors.append(f"清空旧数据库失败: {str(e)}")
        
        if progress_callback:
            progress_callback({
                "status": "saving",
                "message": "正在写入向量数据...",
                "progress": 80
            })
        
        # 写入新数据
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=persist_dir
        )
        
        if progress_callback:
            progress_callback({
                "status": "saving",
                "message": "向量数据写入完成，正在验证...",
                "progress": 95
            })
        
        # 验证
        verify_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        verify_data = verify_store.get()
        result["final_chunks"] = len(verify_data.get('ids', []))
    
    end_time = time.time()
    result["end_time"] = end_time
    result["total_time"] = end_time - start_time
    result["success"] = True
    result["message"] = f"完成！总耗时 {result['total_time']:.1f} 秒"
    
    if progress_callback:
        progress_callback({
            "status": "complete",
            "message": result["message"],
            "progress": 100,
            "summary": result
        })
    
    return result


if __name__ == "__main__":
    # 测试扫描功能
    result = scan_data_directory("./data/")
    print(f"扫描完成: {result['total_files']} 个文件")
    for d in result["directories"]:
        print(f"  {d['name']}: {d['count']} 个文件")
