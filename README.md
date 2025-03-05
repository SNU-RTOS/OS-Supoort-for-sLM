# OS-Supoort-for-on-device-LLM

## Quick start

### 1. Download Bazelisk

This project requires Bazelisk, a version control tool for Bazel. Bazelisk automatically downloads and runs the appropriate Bazel version based on the `.bazelversion` file in your project.

If Bazelisk is not installed, use one of the following methods to install it:

#### (1) Ubuntu/Linux
```sh
curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o /usr/local/bin/bazelisk
chmod +x /usr/local/bin/bazelisk
ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel  # Set up to use 'bazel' command
```

#### (2) macOS (Homebrew)
```sh
brew install bazelisk
```

#### (3) Windows (Scoop)
```sh
scoop install bazelisk
```

#### Downloading a Specific Bazel Version

Bazelisk automatically downloads the Bazel version specified in the `.bazelversion` file in your project's root directory.

- If the `.bazelversion` file is absent, Bazelisk downloads the latest stable version of Bazel.
- If the file is present, Bazelisk downloads and runs the specified version.


### 2. Download external sources and build binary
Run the following command to build the `text_generator_main`:

```sh
./setup.sh
```

Once setup.sh is done, you can build with `build.sh` in the future

```sh
./build.sh
./text_generator_main
```