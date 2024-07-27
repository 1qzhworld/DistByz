@echo off
setlocal enabledelayedexpansion

rem Config files
set CONFIG_FILES= ^
test_209_10H28M31S_blk2_wtAV_init10_ones20_byz5_N50L10D9ss1e1.yml
test_209_10H28M31S_blk2_wtAV_init10_ones100_byz5_N50L10D9ss1e1.yml
test_209_10H28M31S_blk2_wtAV_init10_ones100_byz5_N50L10D9ss5e2.yml
test_209_10H28M31S_blk3_wtAV_init10_ones9_byz5_N50L10D9ss1e1.yml


rem Loop through all config files
for %%f in (%CONFIG_FILES%) do (
    echo ---------------------------------------------------------
    echo Running config file: %%f
    echo ---------------------------------------------------------
    python main.py -a %%f
)

endlocal
pause