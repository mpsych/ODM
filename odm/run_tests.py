import configparser
import os
import shutil
import subprocess

def run_tests():
    # Define the repository and the local directory to clone it into
    repo = "https://github.com/RyanZurrin/ODM_TEST_DATA.git"
    local_dir = "/tmp/ODM_TEST_DATA"

    # Backup the current __configloc__.py file
    shutil.copyfile('core/__configloc__.py', '__configloc__.py.bak')

    try:
        # Clone the repository
        subprocess.check_call(["git", "clone", repo, local_dir])

        # Read the configuration file
        config = configparser.ConfigParser()
        config.read("config/test_config.ini")

        # Update the data_root in the configuration file
        config["5BHIST"]["data_root"] = os.path.join(local_dir, "TEST_DATA")

        # Write the updated configuration to a new file
        updated_config_path = "config/test_config.ini"
        with open(updated_config_path, "w") as configfile:
            config.write(configfile)

        # Update the __configloc__.py file
        with open('core/__configloc__.py', 'w') as f:
            f.write(f'CONFIG_LOC = "{updated_config_path}"')

        # Run the config
        subprocess.check_call(["python", "run_pipeline.py"])
        print("Tests completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print("Tests failed with error code", e.returncode)
        return False

    finally:
        # Restore the original __configloc__.py file
        shutil.copyfile('__configloc__.py.bak', 'core/__configloc__.py')

        # Delete the cloned repository
        shutil.rmtree(local_dir)

        # delete the backup __configloc__.py file
        os.remove('__configloc__.py.bak')


if __name__ == "__main__":
    run_tests()
