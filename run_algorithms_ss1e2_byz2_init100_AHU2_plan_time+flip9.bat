@echo off
setlocal enabledelayedexpansion

set PYTHONPATH=D:\PhD_Qu_from_2021\Sketches\2024-07_BlockByz(code)\BlockByz07_v1

:: Define the configurations
set CONFIG_FILES=^
.\configs\test_2024-07-19_end700_ss1e2_net17_byz2_plantime9_dim10_blksz3_init100_AHU2_bb.yml ^
.\configs\test_2024-07-19_end700_ss1e2_net17_byz2_plantime9_dim10_blksz3_init100_AHU2_bcgd.yml ^
.\configs\test_2024-07-19_end700_ss1e2_net17_byz2_plantime9_dim10_init100_AHU2_dgd.yml ^
.\configs\test_2024-07-19_end700_ssdim_net17_byz2_plantime9_dim10_init100_AHU2_dgd.yml ^
.\configs\test_2024-07-19_end700_ss1e2_net17_byz2_planflip9_dim10_blksz3_init100_AHU2_bb.yml ^
.\configs\test_2024-07-19_end700_ss1e2_net17_byz2_planflip9_dim10_blksz3_init100_AHU2_bcgd.yml ^
.\configs\test_2024-07-19_end700_ss1e2_net17_byz2_planflip9_dim10_init100_AHU2_dgd.yml ^
.\configs\test_2024-07-19_end700_ssdim_net17_byz2_planflip9_dim10_init100_AHU2_dgd.yml

:: Check configurations
for %%f in (%CONFIG_FILES%) do (
    echo Checking configuration %%f
    python .\utils\utils.py --check-config %%f
    if errorlevel 1 (
        echo Configuration file %%f does not match the filename. Exiting.
        exit /b 1
    )
)

:: Execute algorithms based on the configuration file names
for %%f in (%CONFIG_FILES%) do (
    setlocal enabledelayedexpansion
    set filename=%%~nf

    :: Determine the algorithm based on the presence of specific substrings in the filename
    echo ------------- Processing %%f -------------
    echo Filename: !filename!

    set processed=0

    echo !filename! | findstr /i "_bb" >nul
    if not errorlevel 1 (
        echo Running BB algorithm for %%f
        python .\algorithms\bb_algorithm.py -o %%f
        set processed=1
    )

    if !processed! equ 0 (
        echo !filename! | findstr /i "_bcgd" >nul
        if not errorlevel 1 (
            echo Running BCGD algorithm for %%f
            python .\algorithms\bcgd_algorithm.py -o %%f
            set processed=1
        )
    )

    if !processed! equ 0 (
        echo !filename! | findstr /i "_dgd" >nul
        if not errorlevel 1 (
            echo Running DGD algorithm for %%f
            python .\algorithms\dgd_algorithm.py -o %%f
            set processed=1
        )
    )

    if !processed! equ 0 (
        echo Unknown algorithm in file %%f. Skipping.
    )

    endlocal
)

echo --------------------------------------------------------------

endlocal
exit /b