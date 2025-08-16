# Plant-disease-detector

🌿 Plant Disease Detection

A deep learning web app that classifies plant leaves as healthy or diseased.
Built with TensorFlow/Keras and deployed using Flask.

📂 Dataset

Used the New Plant Diseases Dataset from Kaggle.

38 classes including crops like Apple, Corn, Grape, Tomato, etc.

⚙️ Model

EfficientNetB0 with transfer learning (pretrained on ImageNet).

Data augmentation (flip, rotation, zoom).

Trained and saved as plant_disease_model.h5.

🚀 Usage

Install dependencies:

pip install -r requirements.txt


Run the app:

python app.py


Open http://127.0.0.1:5000/ to upload and test images.

📝 Learnings

Applied transfer learning for the first time.

Learned how pretrained models improve accuracy.

Deployed a DL model with Flask.

## 🔗 Model File

The trained model (`plant_disease_model.h5`) is too large for GitHub.  
You can download it from [Google Drive] https://drive.google.com/file/d/1Bz66_UIzx3sWufgP-WES78bk9Q8YLGYt/view?usp=drive_link  

Place it in the project root folder before running `app.py`.

