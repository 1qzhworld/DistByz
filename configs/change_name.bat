@echo off
setlocal enabledelayedexpansion

:: 遍历当前文件夹中的所有文件
for %%f in (*.*) do (
    set "filename=%%~nf"
    set "extension=%%~xf"

    :: 检查文件名是否包含 "end2000"
    echo "!filename!" | findstr /i "end2000" >nul
    if not errorlevel 1 (
        :: 将 "end2000" 替换为 "end1000"
        set "newfilename=!filename:end2000=end1000!"
        echo Renaming "%%f" to "!newfilename!!extension!"
        ren "%%f" "!newfilename!!extension!"
    )
)

endlocal