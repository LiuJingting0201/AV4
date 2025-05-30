Comprehensive autonomous vehicle data processing platform that integrates multiple modules for sensor data fusion, visualization, analysis, and autonomous driving algorithm development. Provides end-to-end solutions for processing LiDAR, Radar, and camera data from autonomous driving datasets, with specialized support for TruckScenes dataset.

 This project is built upon and extends the following official resources:
- **[TruckScenes Official Tutorial](https://github.com/TUMFTM/truckscenes-devkit/blob/main/tutorials/truckscenes_tutorial.ipynb)** - Official TruckScenes development kit tutorial
- **[MAN Brand Portal Dataset](https://brandportal.man/d/QSf8mPdU5Hgj)** - Official TruckScenes dataset access and documentation

- #### Prerequisites
```bash
python >= 3.9
torch >= 1.9.0
torchvision >= 0.10.0
numpy >= 1.21.0
opencv-python >= 4.5.0
open3d >= 0.13.0
plotly >= 5.0.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
truckscenes-devkit

#### Installation
```bash
# Clone the repository
git clone https://github.com/LiuJingting0201/AV4.git
cd AV4

# Create virtual environment
python -m venv av4_env
source av4_env/bin/activate  # On Windows: av4_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install TruckScenes devkit
pip install truckscenes-devkit

# Download pre-trained models
python tools/download_models.py

# Download TruckScenes dataset from MAN Brand Portal
# Visit: https://brandportal.man/d/QSf8mPdU5Hgj
# Extract dataset to ./data/man-truckscenes/
