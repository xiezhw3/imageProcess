rm src/imageProcess
cmake .
make

for i in `ls data`
do
	src/imageProcess "data/$i"
done
