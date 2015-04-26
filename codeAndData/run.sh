rm src/imageProcess
cmake .
make

for i in `ls image`
do
	src/imageProcess "image/$i"
done
