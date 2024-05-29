project('launch_in_startup', 'cpp',
        version : '1.0.0',
        default_options : ['werror=true', 'warning_level=3', 'cpp_std=c++17'])
cpp_arguments = ['-pedantic', '-Wno-unused-parameter', '-faligned-new']
add_project_arguments(cpp_arguments, language : 'cpp')
launch_in_startup = executable('launch_in_startup', 'main.cpp', install : true)

test('test', launch_in_startup)