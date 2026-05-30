"""
Dashboard generation layer.

A thin orchestration layer ABOVE the existing query runtime (retrieval ->
planner -> DataFusion -> agent). It decomposes one natural-language dashboard
prompt into many analytical intents, reuses run_agent_query() per intent to
produce grounded datasets, recommends a visualization per dataset, and
assembles a persisted, render-ready DashboardConfig.

See response.txt for the full architecture. This package contains NO query
logic — all SQL/join/schema work is delegated to the existing agent.

Modules:
  component_catalog       - metadata registry of reusable dashboard components
  data_catalog            - read projection over existing metadata tables
  query_engine            - prompt decomposition + per-widget agent calls + profiling
  recommendation_engine   - dataset shape -> best component + column binding
  assembly_engine         - resolved widgets -> DashboardConfig
"""
