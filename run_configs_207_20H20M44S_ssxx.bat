@echo off
setlocal enabledelayedexpansion

rem Config files
set CONFIG_FILES= ^
test_207_20H20M44S_end300_ss1e1_blk2_wtBB_init10_rand9_net20_byz1_dim4_AHU2_b20_new.yml ^
test_207_20H20M44S_end300_ss1e1_blk2_wtBB_init10_rand9_net20_byz1_dim4_AHU10_b20_new.yml ^
test_207_20H20M44S_end300_ss1e2_blk2_wtBB_init10_rand9_net20_byz1_dim4_AHU2_b20_new.yml ^
test_207_20H20M44S_end300_ss1e2_blk2_wtBB_init10_rand9_net20_byz1_dim4_AHU10_b20_new.yml ^
test_207_20H20M44S_end300_ss1e3_blk2_wtBB_init10_rand9_net20_byz1_dim4_AHU10_b20_new.yml
rem Loop through all config files
for %%f in (%CONFIG_FILES%) do (
    echo ---------------------------------------------------------
    echo Running config file: %%f
    echo ---------------------------------------------------------
    python main.py -a %%f
)

endlocal
pause