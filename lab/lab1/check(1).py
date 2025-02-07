import zipfile
import os
import yaml

def check_zip_file(zip_file_path):
    files_to_check = ["search.py", "searchAgents.py", "multiAgents.py", "info.yaml"]
    missing_files = []

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        find_files = {}
        
        for file in zip_ref.namelist():
            components = file.split("/")
            if len(components) > 2:
                print("压缩包层数过深")
                continue
            find_files[components[-1]] = zip_ref.read(file)
        
        for file_to_check in files_to_check:
            if file_to_check not in find_files:
                missing_files.append(file_to_check)
    
    return missing_files, find_files

def main():
    zip_file_path = input("请输入要检查的 zip 文件路径: ")

    if not os.path.isfile(zip_file_path):
        print("错误：指定的文件路径无效或文件不存在！")
        return

    missing_files, find_files = check_zip_file(zip_file_path)
    
    if missing_files:
        print("缺少以下文件:")
        for file in missing_files:
            print(file)
    else:
        info_yaml = yaml.safe_load(find_files["info.yaml"])
        if info_yaml['student_id'] == 123455 or info_yaml['name'] == '':
            print(f"请修改缺省的学号和姓名: {info_yaml}")
        else:
            print("该 zip 文件包含所有必需的文件。")

if __name__ == "__main__":
    main()
