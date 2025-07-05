# AI-Powered Diet Planner

# Objective
1. Our application, "Snap & Snack" application aims to address the issues with AI-driven image recognition​
2. It automates food identification and nutritional analysis from photos​
3. Users receive personalized insights without the hassle of manual input​
4. This promotes healthier choices and simplifies nutrition management​
5. While in the case of maintaining a healthy diet is difficult due to the effort required in manual food logging, most users lack the expertise to accurately analyze nutritional content​
6. Inaccurate portion estimates lead to poor eating habits and unbalanced diets​
7. Traditional diet tracking methods are time-consuming and often skipped​
8. Thus, our application redefines nutrition tracking by making healthy eating simple, smart, and accessible—turning every food photo into a step toward better health​

# Dataset used
1. The Food-101 dataset is a benchmark dataset in computer vision, particularly designed forfood image classification tasks​
2. It consists of 101 distinct food categories​, 101,000 labeled images (1,000 images per category)​
3. Training set: 750 images per class​
4. Test set: 250 images per class​

# Characteristic of the dataset​
1. Large-scale: Provides enough data for training deep models​
2. Diverse and realistic: Photos include varying backgrounds, angles, lighting conditions - mirroring real-life use cases​
3. Well-labeled: Consistent naming and structured data enable effective supervised learning​

# Tech stack
1. Python – Core language for backend logic, machine learning, and API integration​
2. Streamlit – For building the interactive web interface​
3. Computer Vision - Using Deep Learning models (CNNs) for food classification​ and model possibly trained or fine-tuned using a pre-existing architecture. Eg: ResNet​
4. Hugging Face – For deploying the trained food classifier model for API-based inference​
5. A public Food Image Dataset (here, the "Food-101" dataset), for training the classifier

# Implementation
1. Data Collection & Preprocessing​
  1. The project begins with collecting a comprehensive food image dataset - "Food-101 dataset​
  2. Images are preprocessed by resizing, normalizing pixel values, and augmenting data to improve model robustness​

2. Model Training – Food Classification​
   1. A Convolutional Neural Network (CNN) is used for food classification​
   2. Popular architectures like ResNet are either trained from scratch or fine-tuned with transfer learning​
   3. The model learns to classify food images into predefined categories based on visual features​
      
3. Model Deployment​
   1.The trained model is exported and deployed using Hugging Face for accessible and scalable inference​
   2. It provides a REST API that can classify food images in real time when integrated with the frontend

4. Building the Frontend – Streamlit App​
  1. The user interface is developed using Streamlit, enabling users to upload images and view results instantly​
  2. The app runs on a local Ubuntu environment, acting as the bridge between user inputs and
backend AI services​

5. Nutritional Analysis & Portion Estimation​
  1. Once food is classified, nutritional data (calories, macronutrients) is fetched using either a nutrition database or API​
  2. Portion size is estimated using metadata (e.g., image size, reference objects), which helps in calculating the calorie intake and nutritional values​

6. Health Insights​
  1. We have created several features for the health goals, namely, weight loss, maintenance, muscle gain​
  2. Based on this, the app provides tailored feedback—like exceeding sugar intake, protein deficiency, or meal balance

# Accuracy of the model
1. Achieved a Top-1 classification accuracy of 90+% on the Food-101 dataset​
2. The model used (SigLIP fine-tuned on Food-101) performed with high precision on real-world food images​
3. Accuracy remained consistent across test data due to the model’s robust language-image pretraining​
4. Outperformed baseline models like ResNet-50 and traditional CNNs in food image classification tasks​
5. SigLIP’s multi-modal training enhanced generalization, helping maintain strong accuracy in noisy image conditions​
6. High accuracy makes it reliable for diet planning and food tracking in practical applications​
7. The model has maintained accuracy even with variations in lighting, angles, and presentation food items

# Expected Outcomes
1. The system will successfully identify a wide range of food items from user-uploaded images using deep learning models​
2. ​Users will receive instant and reliable information on calorie content, macronutrients, and portion size without manual entry​
3. Based on individual health goals and profiles, the app will offer tailored dietary suggestions to encourage healthier eating habits​
4. A seamless, intuitive interface (via Streamlit) will encourage users to actively track meals with minimal effort​
5. By visualizing dietary patterns over time, users can better understand their eating habits and make informed lifestyle adjustments​
6. Snap & Snack transforms the tedious task of food logging into a quick, automated experience using image recognition and real-time inference​
7. With model hosting on Hugging Face, the app is scalable and ready for deployment beyond local environments
