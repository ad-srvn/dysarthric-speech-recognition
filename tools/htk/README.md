# HTK installation

HTK is not distributed with this repository. Obtain HTK 3.4.x under its own
license and either:

- copy the required executables into `tools/htk/bin/`;
- add the HTK `bin` directory to `PATH`; or
- set `HTK_BIN` to the HTK `bin` directory.

Required commands: `HCopy`, `HCompV`, `HERest`, `HHEd`, `HParse`, `HVite`, and
`HResults`.

Run `make check` from the repository root after installation. HTK source and
binaries in this directory are ignored by Git.
