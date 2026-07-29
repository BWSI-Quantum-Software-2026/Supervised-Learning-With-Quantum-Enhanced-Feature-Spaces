# same kernel method as run_kernel_adhoc but this is just on iris instead. 
# just a note again that iris wasnt built around our feature map so whatever we get here is real
from src.datasets import load_iris_2feature
from src.kernel.kernel_matrix import compute_kernel_matrix
from src.kernel.classify import train_svm, predict_svm


