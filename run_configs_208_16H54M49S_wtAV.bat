@echo off
setlocal enabledelayedexpansion

rem Config files
set CONFIG_FILES= ^
test_208_16H54M49S_end100_ss1e1_blk3_wtAV_init0_ones9_net50_byz1_dim9_AHU1003_b20.yml ^
test_208_16H54M49S_end100_ss1e1_blk3_wtAV_init0_rand9_net50_byz1_dim9_AHU10_b20.yml ^
test_208_16H54M49S_end100_ss1e1_blk3_wtAV_init10_ones9_net50_byz1_dim9_AHU1003_b20.yml
@REM test_208_16H54M49S_end100_ss1e1_blk3_wtAV_init10_rand9_net50_byz1_dim9_AHU10_b20.yml

rem Loop through all config files
for %%f in (%CONFIG_FILES%) do (
    echo ---------------------------------------------------------
    echo Running config file: %%f
    echo ---------------------------------------------------------
    python main.py -a %%f
)

endlocal
pause