@echo off
echo ============================================
echo   Start Local RAG WebUI (Ollama + LangChain)
echo ============================================

REM 切换到脚本所在目录
cd /d %~dp0

REM 使用 venv 的 python.exe 启动 Streamlit（绕过损坏的 streamlit.exe）
D:\github\.venv\Scripts\python.exe -m streamlit run webui.py

pause