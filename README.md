# CaptchaBypass
CGAN + OCR models designed to generate captcha images along with character recognition in existing captcha. 

### Setup: 
Install required libraries 

```bash
pip install -r requirements.txt
```
### Change file path for dataset variable to your local dataset file path. 

Models were trained on 256x256 images which can be found at: 
https://www.kaggle.com/datasets/akashguna/large-captcha-dataset

For CGAN.py 
```bash
dataset = CustomDataset("your file path", transform=transform)
```

For OCRmodel.py
```bash
dataset = CaptchaDataset("your file path")
```




