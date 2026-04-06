import os
import sys
import subprocess
import platform

def run(cmd):
    """Run a shell command and exit on failure."""
    print(f"\n👉 Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("🚀 ZERO-DAY DETECTION SETUP")
    print("=" * 60)

    py_version = sys.version_info
    print(f"🐍 Python version: {py_version.major}.{py_version.minor}")

    # Enforce Python version
    if py_version.major != 3 or py_version.minor not in [10, 11]:
        print("❌ ERROR: Use Python 3.10 or 3.11 ONLY")
        sys.exit(1)

    # Create virtual environment if missing
    if not os.path.exists("venv"):
        print("📦 Creating virtual environment...")
        run(f"{sys.executable} -m venv venv")
    else:
        print("📦 Virtual environment already exists")

    # Platform-specific paths
    if platform.system() == "Windows":
        pip_path = os.path.join("venv", "Scripts", "pip")
        python_path = os.path.join("venv", "Scripts", "python")
        activate_cmd = "venv\\Scripts\\activate"
    else:
        pip_path = os.path.join("venv", "bin", "pip")
        python_path = os.path.join("venv", "bin", "python")
        activate_cmd = "source venv/bin/activate"

    # Upgrade packaging tools in venv
    print("\n⬆️ Upgrading pip, setuptools, wheel in venv...")
    run(f"{python_path} -m pip install --upgrade pip setuptools wheel")

    # Install dependencies from requirements.txt
    if os.path.exists("requirements.txt"):
        print("\n📦 Installing/upgrading dependencies from requirements.txt...")
        run(f"{python_path} -m pip install --upgrade -r requirements.txt")
    else:
        print("⚠️ No requirements.txt found — skipping dependency install")

    # macOS-specific TensorFlow support
    if platform.system() == "Darwin":
        print("\n🍎 macOS detected — installing TensorFlow Metal support...")
        run(f"{python_path} -m pip install --upgrade tensorflow-macos==2.15.0 tensorflow-metal")

    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)

    print(f"\n👉 Activate your virtual environment with:\n{activate_cmd}")
    print("\n👉 Run your app with:\npython app.py")
    print("\n💡 Or run your app directly inside venv without activating:")
    print(f"{python_path} app.py")

if __name__ == "__main__":
    main()