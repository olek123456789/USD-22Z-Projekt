# USD-22Z-Projekt

Wykorzystywany system: Ubuntu 22.10
Wykorzystywana wersja Mujoco: 2.1.0

## Instrukcja stworzenia wirtualnego środowiska oraz instalacji wymaganych bibliotek

### Instalacja Mujoco
Pobranie Mujoco z github, stworzenie ukrytego folderu katalogu domowym i umieszczenie plików Mujoco w tym katalogu:
```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
tar -xf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

### Dodanie odpowiednich zmiennych środowiskowych do `bashrc`
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/t/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PATH="$LD_LIBRARY_PATH:$PATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### Instalacja bibliotek w systemie Ubuntu 22.10
```
sudo apt install patchelf python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev  libglew-dev python3-pip libosmesa6-dev libgl1-mesa-glx libglfw3
```

### Stworzenie środowiska
```
conda create --name t python=3.7
conda activate t
pip install -r requirements.txt
```