@echo off
REM One-click setup + launch (Windows). Double-click or run from the project folder.
cd /d "%~dp0"
python -m pip install -r requirements-project.txt || goto :err
python -m spacy download en_core_web_sm
if not exist artifacts\models\best_model.pkl python -m src.pipeline || goto :err
python -m streamlit run app\customer_streamlit.py
goto :eof
:err
echo Something failed - check the message above.
pause
