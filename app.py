import subprocess
import sys


def main():
    demo = "--demo" in sys.argv

    if demo:
        print("Initialising database with demo data...")
        subprocess.run(
            [sys.executable, "scripts/init_database.py", "--seed", "--nodes", "9"],
            check=True,
        )
        print("Generating city graph...")
        subprocess.run(
            [sys.executable, "scripts/generate_graph.py", "--rows", "3", "--cols", "3"],
            check=True,
        )

    print("Starting Traffic Digital Twin...")
    subprocess.run(
        [sys.executable, "main.py", "--services", "api", "dashboard"],
        check=False,
    )


if __name__ == "__main__":
    main()
