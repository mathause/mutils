Changelog
=========

v0.4.2 (26.02.2018)
-------------------
- imports maps

v0.4.1 (26.02.2018)
-------------------
- typo import version
- add readme

v0.4.0 (26.02.2018)
-------------------
- add __version__
- plot
  - map_ticks
  - map_gridlines
  - get_map_layout
  - set_map_layout
- clm
  - plot_clm_layers
  - get_root_fraction
  - sm_weights_monthly, plot_sm_weights_monthly git and example_sm_weights_monthly
- geo
  - CosWgt*
- maps
  - pcolormesh*
- postprocess
  - _maybe_recompute\_ (check if dest file exists)**
- stats
  - adjust_alpha* (use statsmodels functionality instead)
  - return_time


* not sure if used anywhere - should be deprecated
** incomplete functionality

v0.3.0
------
- add xray and additional xray functions
- plot: add get_months
- water_vapour: better T bounds check
- water_vapour: add psychrometric_const

v0.2.0
------
- water_vapour module
- various conversion function for water vapor in the atmosphere
- some tests for water vapor module

v0.1.0
------
- format time x-axis with months
- clear sky index (Marty and Philipona)
- rgpot: potential radiation at earth surface in absence of atmosphere














