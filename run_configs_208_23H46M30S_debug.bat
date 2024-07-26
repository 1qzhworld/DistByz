@echo off
setlocal enabledelayedexpansion

rem Config files
set CONFIG_FILES= ^
test_208_23H46M30S_end10_debug.yml


rem Loop through all config files
for %%f in (%CONFIG_FILES%) do (
    echo ---------------------------------------------------------
    echo Running config file: %%f
    echo ---------------------------------------------------------
    python main.py -a %%f
)

endlocal
pause