@echo off
setlocal enabledelayedexpansion

rem Config files
set CONFIG_FILES= ^
test_208_11H45M33S_end100_ss1e2_blk1_wtBB_init10_rand9_net50_byz1_dim9_AHU100_b20.yml ^
test_208_11H45M33S_end100_ss1e2_blk2_wtBB_init10_rand9_net50_byz1_dim9_AHU100_b20.yml ^
test_208_11H45M33S_end100_ss1e2_blk3_wtBB_init10_rand9_net50_byz1_dim9_AHU100_b20.yml ^
test_208_11H45M33S_end100_ss1e2_blk4_wtBB_init10_rand9_net50_byz1_dim9_AHU100_b20.yml

rem Loop through all config files
for %%f in (%CONFIG_FILES%) do (
    echo ---------------------------------------------------------
    echo Running config file: %%f
    echo ---------------------------------------------------------
    python main.py -a %%f
)

endlocal
pause