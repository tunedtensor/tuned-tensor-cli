@echo off
"%~dp0runtime\node.exe" "%~dp0node_modules\@tuned-tensor\cli\dist\index.js" %*
exit /b %errorlevel%
