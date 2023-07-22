#!/usr/bin/env python3
# import configparser
# import os
# import shutil
# import subprocess
# import tempfile
#
#
# def run_tests():
#     # Define the repository and the local directory to clone it into
#     repo = "https://github.com/RyanZurrin/ODM_TEST_DATA.git"
#
#     # Backup the current __configloc__.py file
#     shutil.copyfile("core/__configloc__.py", "__configloc__.py.bak")
#
#     with tempfile.TemporaryDirectory() as local_dir:
#         try:
#             return _run_test_driver(repo, local_dir)
#         except subprocess.CalledProcessError as e:
#             print(f"Tests failed with error code {e.returncode}")
#             return False
#         finally:
#             # Restore the original __configloc__.py file
#             shutil.copyfile("__configloc__.py.bak", "core/__configloc__.py")
#
#             if os.path.isdir('/tmp/odm_test/'):
#                 shutil.rmtree('/tmp/odm_test/')
#
#             # delete the backup __configloc__.py file
#             os.remove("__configloc__.py.bak")
#
#
# def _run_test_driver(repo, local_dir):
#     """
#     """
#     subprocess.check_call(["git", "clone", repo, local_dir])
#
#     # Read the configuration file
#     config = configparser.ConfigParser()
#     config.read("config/test_config.ini")
#
#     # Update the data_root in the configuration file
#     config["5BHIST"]["data_root"] = os.path.join(local_dir, "TEST_DATA")
#
#     # Write the updated configuration to a new file
#     updated_config_path = "config/test_config.ini"
#     with open(updated_config_path, "w") as configfile:
#         config.write(configfile)
#
#     # Update the __configloc__.py file
#     with open("core/__configloc__.py", "w") as f:
#         f.write(f'CONFIG_LOC = "{updated_config_path}"')
#
#     # Run the config
#     subprocess.check_call(["python", "run_pipeline.py"])
#
#     # Count the number of images
#     with open('/tmp/odm_test/logs/good_paths.txt', 'r') as f:
#         good_images_count = sum(1 for _ in f)
#
#     if good_images_count == 81:
#         print("Tests completed successfully!")
#         return True
#     else:
#         print(f"Expected 81 images, but found {good_images_count}")
#         return False
#
#
#
# if __name__ == "__main__":
#     run_tests()




import configparser
import os
import shutil
import subprocess
import tempfile

print(f"Current working directory: {os.getcwd()}")
def run_tests():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the repository and the local directory to clone it into
    repo = "https://github.com/RyanZurrin/ODM_TEST_DATA.git"

    # Backup the current __configloc__.py file
    configloc_src = os.path.join(current_dir, "core", "__configloc__.py")
    configloc_bak = os.path.join(current_dir, "__configloc__.py.bak")
    shutil.copyfile(configloc_src, configloc_bak)

    with tempfile.TemporaryDirectory() as local_dir:
        try:
            return _run_test_driver(repo, local_dir)
        except subprocess.CalledProcessError as e:
            print(f"Tests failed with error code {e.returncode}")
            return False
        finally:
            # Restore the original __configloc__.py file
            shutil.copyfile(configloc_bak, configloc_src)

            if os.path.isdir('/tmp/odm_test/'):
                shutil.rmtree('/tmp/odm_test/')

            # delete the backup __configloc__.py file
            os.remove(configloc_bak)


def _run_test_driver(repo, local_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    subprocess.check_call(["git", "clone", repo, local_dir])

    # Read the configuration file
    config = configparser.ConfigParser()
    test_config = os.path.join(current_dir, "config", "test_config.ini")
    config.read(test_config)

    # Update the data_root in the configuration file
    config["5BHIST"]["data_root"] = os.path.join(local_dir, "TEST_DATA")

    # Write the updated configuration to a new file
    updated_config_path = os.path.join(current_dir, "config", "test_config.ini")
    with open(updated_config_path, "w") as configfile:
        config.write(configfile)

    # Update the __configloc__.py file
    with open(os.path.join(current_dir, "core", "__configloc__.py"), "w") as f:
        f.write(f'CONFIG_LOC = "{updated_config_path}"')

    # Run the config
    subprocess.check_call(["python", os.path.join(current_dir, "run_pipeline.py")])

    # Count the number of images
    with open('/tmp/odm_test/logs/good_paths.txt', 'r') as f:
        good_images_count = sum(1 for _ in f)

    if good_images_count == 81:
        print("Tests completed successfully!")
        return True
    else:
        print(f"Expected 81 images, but found {good_images_count}")
        return False

if __name__ == "__main__":
    run_tests()
