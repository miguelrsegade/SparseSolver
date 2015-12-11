file ( GLOB_RECURSE hdr *.hpp )       
file ( GLOB_RECURSE cpp *.cpp )       
file ( GLOB_RECURSE cu *.cu )         
                                      
set ( SOURCES ${hdr} ${cpp} ${cu} )   
                                      
cuda_add_library (pssolver ${SOURCES})
