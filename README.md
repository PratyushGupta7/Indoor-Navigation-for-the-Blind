# Indoor Navigation for the Blind 🧭🦯  
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/PratyushGupta7/Indoor-Navigation-for-the-Blind/blob/main/LICENSE.md) 
[![GitHub issues](https://img.shields.io/github/issues/PratyushGupta7/Indoor-Navigation-for-the-Blind)](https://github.com/PratyushGupta7/Indoor-Navigation-for-the-Blind/issues) 
[![GitHub stars](https://img.shields.io/github/stars/PratyushGupta7/Indoor-Navigation-for-the-Blind)](https://github.com/PratyushGupta7/Indoor-Navigation-for-the-Blind/stargazers)


A client-server assistive system that helps visually impaired individuals navigate indoor environments using computer vision. The system combines YOLOv5s for object detection, MiDaS DPT_Hybrid for depth estimation, and ORB visual odometry to deliver real-time audio guidance through a web or mobile application.

## 📋 Table of Contents  
- [🎯 Problem Statement](#-problem-statement)  
- [💡 Motivation](#-motivation)  
- [✨ Key Features of our approach](#-key-features-of-our-approach)  
- [🏗️ System Architecture](#-system-architecture)  
- [🚀 Installation & Setup](#-installation--setup)  
  - [Backend Setup](#backend-setup)  
  - [Frontend Setup](#frontend-setup)  
- [🚦 Usage](#-usage)  
- [📊 Performance Metrics](#-performance-metrics)  
- [📈 Future Work](#-future-work)  
- [👥 Contributors](#-contributors)  
- [🙏 Acknowledgements & Key References](#-acknowledgements--key-references)  
- [📜 License](#-license)



## 🎯 Problem Statement

Developing effective navigational assistance for visually impaired individuals in indoor environments is a significant challenge due to the complex, variable nature of these spaces and the critical need for reliable, real-time perception and guidance. This project aims to create a scalable, vision-only indoor navigation aid that delivers natural language output by addressing the following core challenges:

-   **Spatial and Appearance Variability:** Indoor environments exhibit diverse layouts, textures, lighting conditions, and clutter, which can degrade the performance of detection and mapping algorithms.
-   **GPS-Denied Localization:** The absence of reliable GPS signals indoors necessitates robust vision-based Simultaneous Localization and Mapping (SLAM) or visual odometry solutions that can provide accurate pose estimates over extended trajectories with minimal drift.
-   **Depth Estimation Reliability:** Monocular depth cues can be noisy and ambiguous, especially in low-texture areas, under poor lighting conditions, or in cluttered scenes.
-   **Visual Odometry Drift:** Frame-to-frame visual odometry tends to accumulate errors over time, especially without loop closures or global corrections, leading to degraded localization accuracy.
-   **Real-Time Computational Constraints:** Executing complex deep learning models for object detection and depth estimation at interactive frame rates (e.g., <100 ms/frame) on mobile or even powerful backend GPUs requires highly optimized inference pipelines.
-   **Instruction Clarity vs. Cognitive Load:** Navigational instructions delivered in natural language must strike a balance between being informative and concise to avoid overwhelming users, particularly in complex environments.
-   **Seamless Multi-Module Integration:** The vision, planning, and text-to-speech components must interoperate efficiently, especially on resource-constrained hardware, without introducing perceptible latency that could hinder real-time guidance.

## 💡 Motivation

The primary motivations driving this project are centered on enhancing the independence and safety of visually impaired individuals in indoor environments:

-   **Accessibility:** To develop an affordable navigation solution that relies solely on a camera, leveraging widely available hardware such as smartphones or AR glasses, thus avoiding the need for expensive specialized equipment.
-   **Scalability:** To create a system that does not depend on pre-installed infrastructure (e.g., beacons, Wi-Fi fingerprinting, or LiDAR scans of the environment), enabling straightforward deployment and use in a wide variety of previously unmapped indoor settings.
-   **Real-time Guidance:** To ensure that the perception and instruction generation processes operate with low latency, providing users with timely alerts, obstacle warnings, and navigational cues crucial for safe and fluid movement.
-   **User-Centric Design:** To prioritize the specific needs and preferences of visually impaired users by delivering intuitive, non-visual cues (primarily voice prompts and haptic feedback) that effectively translate complex spatial information into simple, actionable guidance, thereby enhancing user autonomy and minimizing cognitive load.

## ✨ Key Features of our approach

- **Real-Time Processing**: ~25 FPS end-to-end on RTX 3080 + i7 CPU  
- **Object Detection**: YOLOv5s (12ms/frame, mAP@0.50: 0.80)  
- **Depth Estimation**: MiDaS DPT_Hybrid (18ms/frame)  
- **Visual Odometry**: ORB + PnP for pose tracking  
- **Path Planning**: A* algorithm with 71% navigation success rate  
- **Accessible Feedback**: Audio instructions via TTS.
- **Client-Server Architecture**: Frontend with FastAPI backend containerized with docker

## 🏗️ System Architecture

The system uses a client-server model where:

- **Frontend Client** captures camera frames and delivers audio feedback  
- **FastAPI Backend** processes frames through parallel CV pipelines:  
  - Object detection (YOLOv5s)  
  - Depth estimation (MiDaS)  
  - Visual odometry (ORB features)  
- **A\* Path Planning** computes safe navigation routes  
- **TTS Engine** converts waypoints to natural language instructions  

![System Architecture](/Report%20and%20PPT/SystemFlow.png)

## 🚀 Installation & Setup

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/PratyushGupta7/Indoor-Navigation-for-the-Blind.git
cd Indoor-Navigation-for-the-Blind/backend
```

#### Option 1: Using Docker (recommended)

```bash
docker build -t indoor-nav-backend .
docker run -p 8000:8000 --gpus all indoor-nav-backend
```

#### Option 2: Manual setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app/main.py
```

### Frontend Setup

The current frontend is a Next.js project bootstrapped with `create-next-app`. An Android app using Kotlin is also under development.

#### Prerequisites:

- Node.js (v16+)  
- npm, yarn, pnpm, or bun

#### Setup Steps:

```bash
cd ../frontend
```

Install dependencies:

```bash
# Choose one
npm install
# or
yarn install
# or
pnpm install
# or
bun install
```

Run Development Server:

```bash
# Choose one
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Access the Application:

- Open `http://localhost:3000` in your browser.  
- You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

#### Additional Notes:

- This project uses `next/font` to optimize and load the Geist font.  
- For deployment, the easiest way is to use the [Vercel Platform](https://vercel.com).  
- Learn more about Next.js from the [Next.js Documentation](https://nextjs.org/docs) or [interactive tutorial](https://nextjs.org/learn).  

**Note on Android App**: We are actively working on building and deploying a Kotlin-based Android app. Updates will be posted here.

## 🚦 Usage

1. Start the backend server ([see Backend Setup](#backend-setup))  
2. Launch the frontend application ([see Frontend Setup](#frontend-setup))  
3. Access the app via `http://localhost:3000`  
4. Ensure the frontend is configured to communicate with the backend (update API endpoints if necessary)  
5. Use the interface to start capturing video input and receive audio guidance for navigation  

## 📊 Performance Metrics

| Component         | Technology         | Performance                        |
|------------------|--------------------|------------------------------------|
| Object Detection | YOLOv5s            | 12ms/frame, mAP@0.50: 0.80         |
| Depth Estimation | MiDaS DPT_Hybrid   | 18ms/frame, AbsRel: 1.2840         |
| Visual Odometry  | ORB + PnP          | 10ms/frame                         |
| Path Planning    | A* Algorithm       | 5ms/plan                           |
| End-to-End       | Client-Server      | ~25 FPS                            |
| Navigation       | Real-world Testing | 71% success rate                   |

## 📈 Future Work

- **Improved Depth Estimation**: Self-supervised learning, stereo refinement  
- **Scale Disambiguation**: VO-based scaling for absolute depth  
- **Model Optimization**: Quantization and pruning  
- **Advanced Scene Understanding**: SpatialLM for layout recognition  
- **Enhanced SLAM**: Mast3r-SLAM for better 3D mapping  
- **Android App Deployment**: Kotlin-based native app in development  


## 👥 Contributors

This project was a collaborative effort, with all team members contributing significantly to the design, development, integration, and testing of the entire pipeline, from the core computer vision modules to the real-time navigation logic and user feedback systems.

-   **Abhishek Bansal** ([abhishek22021@iiitd.ac.in](mailto:abhishek22021@iiitd.ac.in))
-   **Dhruv Sharma** ([dhruv22170@iiitd.ac.in](mailto:dhruv22170@iiitd.ac.in))
-   **Pratyush Gupta** ([pratyush22375@iiitd.ac.in](mailto:pratyush22375@iiitd.ac.in)) - *Repository Owner*
-   **Vinayak Agrawal** ([vinayak22574@iiitd.ac.in](mailto:vinayak22574@iiitd.ac.in))

*(All affiliated with IIITD, Delhi)*

## 🙏 Acknowledgements & Key References

This work builds upon a rich history of research in computer vision, robotics, and assistive technologies. We acknowledge the foundational contributions of researchers in these fields. Some key references that inspired or were utilized in this project include:

-   Davison, A. J., Reid, I. D., Molton, N. D., & Stasse, O. (2007). MonoSLAM: Real-time single camera SLAM. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 29*(6), 1052-1067.
-   Eigen, D., Puhrsch, C., & Fergus, R. (2014). Depth map prediction from a single image using a multi-scale deep network. *Advances in Neural Information Processing Systems, 27*.
-   Jocher, G., et al. (2020). Ultralytics YOLOv5. *GitHub Repository*. [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
-   Mur-Artal, R., & Tardós, J. D. (2015). ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras. *IEEE Transactions on Robotics, 33*(5), 1255-1262.
-   Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 44*(1), 382-393. (MiDaS)
-   Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision Transformers for Dense Prediction. *arXiv preprint arXiv:2103.13413*. (DPT used in MiDaS)

*(A more comprehensive list of references is available in the project's final report.)*

## 📜 License

This project is licensed under the [MIT License](https://github.com/PratyushGupta7/Indoor-Navigation-for-the-Blind/blob/main/LICENSE.md).
