# Vehicle-AccidentYoloApp

## 🚗 About the Project

**Vehicle-AccidentYoloApp** is a **YOLO-based application** used for **vehicle and accident detection** from video or in **real-time**. This application can identify various types of vehicles, calculate vehicle speed, and provide alerts when an accident is detected.

This project is deployed **locally** on the user's device to avoid high costs of renting online GPU resources.

---

## 🖥️ System Requirements

To run this application optimally, the recommended device specifications are:

### **Hardware:**

-   **CPU**: AMD Ryzen 5 5600 (6 Core, 12 Thread, 3.5 GHz - 4.4 GHz)
-   **GPU**: NVIDIA GeForce RTX 3060 (8GB GDDR6, CUDA Core 3584)
-   **RAM**: 16GB (Dual Channel, 3200MHz)
-   **Storage**: SSD NVMe 500GB (PCIe Gen 3x4)
-   **OS**: Windows 11 (version 24H2)

### **Software & Library:**

-   **Python**: 3.12.7
-   **Code Editor**: Visual Studio Code (version 1.96.4)
-   **Required libraries:**
    -   NumPy (2.2.2)
    -   OpenCV (4.10.0.84)
    -   Pandas (2.2.3)
    -   Pillow (11.1.0)
    -   Plotly (5.24.1)
    -   Streamlit (1.40.2)
    -   Supervision (0.25.1)
    -   Torch (2.5.1)
    -   Ultralytics (8.3.40)

---

## 📦 Installation

Follow these steps to install and run the application:

### **1. Clone the Repository**

```sh
git clone https://github.com/rai-pramana/Vehicle-AccidentYoloApp.git
cd Vehicle-AccidentYoloApp
```

### **2. Create a Virtual Environment (Optional, but recommended)**

```sh
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate  # For Windows
```

### **3. Install Dependencies**

```sh
pip install -r requirements.txt
```

Or install them one by one:

```sh
pip install numpy opencv-python pandas pillow plotly streamlit supervision torch ultralytics
```

### **4. Install CUDA**

For NVIDIA CUDA installation, please follow this tutorial: [Setup-NVIDIA-GPU-for-Deep-Learning](https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning)

### **5. Run the Application**

```sh
streamlit run app.py
```

Access the application via browser at **http://localhost:8501**.

---

## 🔧 How to Use

1. **Select detection mode**:
    - **Image Detection**: Processes image input.
    - **Video Detection**: Processes all frames from the input video for more accurate results.
    - **Real-Time Detection**: Uses screen recording (OBS Virtual Camera) for live detection with limited FPS based on device capability.
2. **Input an image, video, or turn on the webcam**.
3. **The application will display detection results, vehicle speed (Video or Real-Time Detection), and accident alerts (Video or Real-Time Detection)** if detected.
4. **Speed and accident data will be recorded in a report (xlsx file)**.

---

## ⚡ Feature Differences: Video vs. Real-Time Detection

-   **Video Detection**:
    -   Processes all frames in the video.
    -   More accurate speed estimation.
    -   Longer processing time.
-   **Real-Time Detection**:
    -   Processes fewer frames (on this device only 11.364 FPS).
    -   Faster due to frame skipping.
    -   Speed estimation accuracy is lower compared to video detection.

---

## 📌 Notes

This application is **deployed locally** because running vehicle and accident detection models requires **high GPU resources**, which would be expensive if done online.

---

## 📷 Screenshot

Here is the application interface:

### **Image Detection Feature**

![Image Detection Feature](assets/Screenshot%20Fitur%20Deteksi%20Gambar.png)

### **Video Detection Feature**

![Video Detection Feature](assets/Screenshot%20Fitur%20Deteksi%20Video.png)

### **Real-Time Detection Feature**

![Real-Time Detection Feature](assets/Screenshot%20Fitur%20Deteksi%20Real-Time.png)

---

## 📜 License

This project uses the **MIT License**. Please use and develop as needed.

---

## 📬 Contact

If you have questions or issues, please contact:

-   **GitHub**: [Vehicle-AccidentYoloApp](https://github.com/rai-pramana/Vehicle-AccidentYoloApp)
-   **Email**: rai.pramana46@gmail.com

---

🚀 Happy trying! Hopefully, this application is useful for effective vehicle and accident detection. 😊
