TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

CUDA_PATH=/usr/local/cuda/
ARCH=sm_37
CXXFLAGS=''

if [[ "$OSTYPE" =~ ^darwin ]]; then
	CXXFLAGS+='-undefined dynamic_lookup'
fi

cd roi_pooling_layer

if [ -d "$CUDA_PATH" ]; then
	echo "GPU"

	TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
	nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
		-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS \
		-arch=$ARCH

	g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		roi_pooling_op.cu.o -I $TF_INC  -D GOOGLE_CUDA=1 -fPIC $CXXFLAGS \
		-L$TF_LIB -ltensorflow_framework -lcudart -L $CUDA_PATH/lib64
else
	echo "CPU"
	TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

	g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		-I $TF_INC -fPIC $CXXFLAGS -L$TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0
fi

cd ..
