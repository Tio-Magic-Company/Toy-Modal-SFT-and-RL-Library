# toy_modal User Documentation

This directory contains user-facing documentation for `toy_modal`.

Start with [`index.md`](index.md). It links to the Modal quickstart, core
concepts, tutorials, cookbook recipes, Modal deployment, operations, reference
pages, and troubleshooting.

The docs are clean-room and Modal-first. Examples use the deployed
`modal-direct` path and assume user-owned Modal resources. Unit tests remain
no-credential by using fake transports and fake Modal SDK objects.

The tutorial docs under [`tutorials/`](tutorials/index.md) are Modal-first
because they accompany the Marimo notebooks in [`tutorials/notebooks/`](tutorials/notebooks/). Those notebooks
keep deployment, GPU work, and publishing behind explicit UI controls.
