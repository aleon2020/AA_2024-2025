# Aprendizaje Automático 2024-2025

¡Bienvenido! En este repositorio se encuentran todos los materiales correspondientes a la asignatura de Aprendizaje Automático.

A continuación se detallan brevemente todos los contenidos que se encuentran en este repositorio, con el objetivo de facilitar la preparación del examen final de la asignatura y de abordar la misma lo mejor posible.

IMPORTANTE: SI OBSERVAS QUE HAY ALGÚN ERROR O ALGO QUE FALTE EN ALGÚN ARCHIVO SUBIDO A ESTE REPOSITORIO (O SI HAY ALGUNA DUDA EN CUANTO A COMPRENSIÓN), DÉJAME UN ISSUE Y TRATARÉ DE RESOLVER EL PROBLEMA LO ANTES POSIBLE. NO TE OLVIDES DEJARME UNA STAR Y ESPERO QUE TODO ESTE MATERIAL TE SEA DE GRAN AYUDA.

## 1. Ejercicios propuestos, prácticas y exámenes

Directorio ['Códigos Libro'](https://github.com/aleon2020/AA_2024-2025/tree/main/C%C3%B3digos%20Libro): Este directorio contiene todo el código fuente que se ve en el libro de la asignatura, además de varios ejemplos que se han visto durante las clases para facilitar la comprensión de los contenidos vistos en teoría (chXX corresponde al código del libro, exXX a ejemplos vistos en clase y prXX a las prácticas).

## 2. Referencias

Directorio ['Referencias'](https://github.com/aleon2020/AA_2024-2025/tree/main/Referencias): Se encuentra el libro que abarca todos los contenidos vistos en las clases de teoría.

## 3. Activación del entorno Conda en los ordenadores de la universidad

PASO 1: Activa Conda introduciendo el siguiente fragmento de texto en el fichero .bashrc de tu HOME.

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/alumnos/USERNAME/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/alumnos/USERNAME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/alumnos/USERNAME/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/alumnos/USERNAME/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

IMPORTANTE: Sustituye USERNAME por tu nombre de usuario de los laboratorios de la universidad.
Algunas de las rutas del fichero PUEDEN variar dependiendo de la ubicacion en la que hayas
instalado Conda.

PASO 2: Guarda los cambios en el fichero .bashrc, cierra y vuelve a abrir una nueva terminal,
para que los cambios queden guardados.

PASO 3: Una vez abierta una nueva terminal, tu prompt deberia aparecer de la siguiente forma.

```sh
(base) USERNAME@f-lXXXX-pcXX:~$
```

PASO 4: Una vez hecho esto, ejecuta los siguientes comandos en la terminal.

```sh
conda config --append channels conda-forge
```

```sh
conda create -n "pyml-book" python=3.9 numpy=1.21.2 scipy=1.7.0 scikit-learn=1.0 matplotlib=3.4.3 pandas=1.3.2
```

```sh
conda activate "pyml-book"
```

PASO 5: Una vez ejecutes este último comando, tu prompt deberia aparecer de la siguiente forma.

```sh
(pyml-book) USERNAME@f-lXXXX-pcXX:~$
```

PASO 6: Instala jupyterlab y las librerías mlxtend y seaborn para poder ejecutar correctamente archivos .ipynb.

```sh
conda install -c conda-forge jupyterlab
```

```sh
conda install mlxtend
```

```sh
conda install seaborn
```

IMPORTANTE: Para instalar las librerías mlxtend y seaborn en tu linux personal, solo tienes que ejecutar los siguientes comandos:

```sh
pip install mlxtend
```

```sh
pip install seaborn
```