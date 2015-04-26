project( src/imageProcess )
find_package( OpenCV REQUIRED )
add_executable( src/imageProcess src/imageProcess )
target_link_libraries( src/imageProcess ${OpenCV_LIBS} )