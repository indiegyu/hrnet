# How to run (test and visualization on SHHA Dataset)

Original repository: https://github.com/taohan10200/IIM

How to run test and visualization:
1. Download these file:
   * PretrainedModels: https://kowondongseoackr-my.sharepoint.com/:f:/g/personal/20215136_office_dongseo_ac_kr/EvpaPdTGe2dOrRUALLa8HssBzgJXpZj_is6K0GHqpb9ydA?e=VNIxbY 
   * ProcessedData: https://kowondongseoackr-my.sharepoint.com/:f:/g/personal/20215136_office_dongseo_ac_kr/EgzuHL5-z_hLqy7FoTgNMEkBgvpJEDLVSFsq9Cq4gIjgXA?e=2D4Ubl 
2. The file structure should be like this:

 ```
    -- ProcessedData
		|-- SHHA
			|-- images
			|   |-- 0001.jpg
			|   |-- 0002.jpg
			|   |-- ...
			|   |-- 0482.jpg
			|-- mask
			|   |-- 0001.png
			|   |-- 0002.png
			|   |-- ...
			|   |-- 0482.png
			|-- train.txt
			|-- val.txt
			|-- test.txt
			|-- val_gt_loc.txt
			|-- test_gt_loc.txt
			|-- ...
	-- PretrainedModels
	  |-- hrnetv2_w48_imagenet_pretrained.pth
	  |-- SHHA-HR-ep_905_F1_0.715_Pre_0.760_Rec_0.675_mae_112.3_mse_229.9.pth
	-- IIM
	  |-- datasets
	  |-- misc
	  |-- ...
 ```

3. Create Anaconda environment with python version 3.8.
4. Install the latest PyTorch (2.2.2) with CUDA 11.8 on the created environment.
5. Run `python test.py`. This will generate `SHHA_HR_Net_test.txt` in `saved_exp_results`. This file contains the image id along with headcounts and its coordinates.
6. Run `python vis4val.py`. This will save visualization images based on the coordinates from `SHHA_HR_Net_test.txt`. The saved images will be saved in `saved_exp_results/SHHA_vis_test_results/`

Note: If you run into an error regarding library/dependancy when you run the code, just google the name of the library and `pip install` the library into the anaconda environment, most of the time it will be fixed without problem.
