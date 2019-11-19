# Calcium scoring in CTs of the chest

This repository contains code for calcium scoring in chest CT scans using deep convolutional neural networks.

The repository contains a jupyter notebook as well as standalone python code (in the subfolder `src`). The notebook can be used to train and
test the networks.

Steps required to train and test the method:
* Create a folder on a linux server with GPU with the following structure:
    * images (folder with images)
    * annotations (folder with corresponding annotations)
    * dataset.csv (see example file)
* Create an empty scratch directory (for any kind of output)
* Build the docker image: `docker build -t calcium-scoring .`
* Run the docker image:
  ```
  docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -p 8888:8888 --rm -it \
      -v /location/of/data/:/home/user/input/:ro \
      -v /location/of/scratch/:/home/user/scratch/:rw \
      calcium-scoring
  ```
* Now use the displayed token to open the notebook in a browser and then open [calcium_scoring_with_convnets](calcium_scoring_with_convnets.ipynb).

## Related publications

* N. Lessmann, I. Isgum, A.A.A. Setio, B.D. de Vos, F. Ciompi, P.A. de Jong, M. Oudkerk, W.P.Th.M. Mali, M.A. Viergever, B. van Ginneken, 
  Deep convolutional neural networks for automatic coronary calcium scoring in a screening study with low-dose chest CT, SPIE Medical Imaging 2016

* N. Lessmann, B. van Ginneken, M. Zreik, P.A. de Jong, B.D. de Vos, M.A. Viergever, I. Isgum. Automatic calcium scoring in low-dose chest 
  CT using deep neural networks with dilated convolutions. IEEE Transactions on Medical Imaging 2018

* S.G.M. van Velzen, N. Lessmann, B.K. Velthuis, I.E.M. Bank, D.H.J.G. van den Bongard, T. Leiner, P.A. de Jong,
    W.B. Veldhuis, A. Correa, J.G. Terry, J.J. Carr, M.A. Viergever, H.M. Verkooijen, I. Išgum.
    Deep learning for automatic calcium scoring in CT: Validation using multiple Cardiac CT and Chest CT protocols. Submitted to Radiology.

## License

GNU General Public License v3.0+ (see LICENSE file or https://www.gnu.org/licenses/gpl-3.0.txt for full text)

Copyright 2019, University Medical Center Utrecht
