## Activación del entorno miniconda en los ordenadores de la universidad

**PASO 1**: Creación de un directorio para Miniconda

Ejecuta el siguiente comando para crear el directorio en el que se instalará miniconda:

```sh
mkdir -p ~/miniconda3
```

**PASO 2**: Descarga del script de instalación de miniconda

Usa el siguiente comando para descargar el instalador de miniconda:

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
```

**PASO 3**: Instalación de miniconda

Ejecuta el script descargado con el siguiente comando:

```sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```

**PASO 4**: Limpieza de archivos de instalación

Ejecuta el siguiente comando para eliminar el script de instalación para ahorrar espacio:

```sh
rm ~/miniconda3/miniconda.sh
```

**PASO 5**: Activación de miniconda

Activa miniconda en tu terminal con el siguiente comando o añade esta línea en tu fichero .bashrc:

```sh
source ~/miniconda3/bin/activate
```

**PASO 6**: Creación del entorno de Python 'pyml-book'

Ejecuta el siguiente comando para crear el entorno 'pyml-book':

```sh
conda create --name pyml-book python==3.9
```

**PASO 7**: Activación del entorno 'pyml-book'

Una vez creado el entorno, actívalo con:

```sh
conda activate pyml-book
```

**PASO 8**: Instalación de librerías

Instala las librerías de aprendizaje automático, visualización de datos y otras dependencias en el entorno:

* Librerías básicas para ciencia de datos:

```sh
conda install numpy scipy scikit-learn matplotlib pandas
```

* Librerías adicionales para visualización y utilidades:

```sh
conda install mlxtend seaborn
```

* Librerías para trabajar con PyTorch:

```sh
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

* JupyterLab para trabajar en notebooks:

```sh
conda install -c conda-forge jupyterlab
```

**PASO 9**: Verificación de la instalación

Para asegurarte de que el entorno está correctamente configurado, puedes listar las librerías instaladas con:

```sh
conda list
```