# Vendored from upstream

Source: https://github.com/lucas-maes/le-wm
Pinned commit: ca231f9f9d9ab041034b6d05e90b6e04bd6cff82
Date: 2026-03-26

Do not edit files in this directory directly. Modifications for integration
live in `subprojects/lewm-integration/liquid_arc_lewm/` and override/monkey-patch
this package from outside.

To refresh the snapshot:
  rm -rf le-wm
  git clone --depth 1 https://github.com/lucas-maes/le-wm.git le-wm
  rm -rf le-wm/.git
