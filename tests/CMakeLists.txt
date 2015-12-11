include_directories (${PROJECT_SOURCE_DIR})                      
file ( GLOB TEST_SOURCES *.cpp )                                 
                                                                 
foreach ( testsourcefile ${TEST_SOURCES} )                       
    get_filename_component ( testname ${testsourcefile} NAME_WE )
    cuda_add_executable( ${testname} ${testsourcefile} )         
    target_link_libraries ( ${testname} pssolver )               
endforeach ( testsourcefile ${TEST_SOURCES} )                    
