from src.core.workflow import Workflow
from src.nodes.connections.sql_connection_node import MSSQLConnectionNode
from src.nodes.io.sql_query_node import MSSQLQueryNode

connection_node = MSSQLConnectionNode(
    name="Database Connection",
    server="PS-UAT-AZS-DWH1",
    database="BIA",
    trusted_connection=True
)

query_node = MSSQLQueryNode(
    name="Business Logic Query",
    description="Query business logic rules",
    execute_mode="get_all",
    connection=connection_node,
    query="""
        SELECT
            [BusinessLogicNo] AS [Id],
            [ProjectName],
            [PromptCategory],
            [ReviewCategory],
            [GuidanceRecommendation],
            [GuidanceJustification],
            [RiskRating],
            [Has_Disclaimer],
            [Has_NewZealand],
            [Has_ShortForm]
        FROM [Cuida].[CREV].[BusinessLogic]
        WHERE [ProjectName] = 'Marketing Review'
        AND [Has_ShortForm] = 0
        AND [BusinessLogicNo] NOT IN (38, 50)
        AND [IsCurrent] = 1
        ORDER BY [BusinessLogicNo]
    """
)

query_workflow = (
    Workflow("Test Workflow")
        .version("1.0")
        .description("Test workflow for querying business logic rules")
        .state("connection", connection_node).start().next("query")
        .state("query", query_node).terminal()
        )

print(query_workflow.get_state_info("query"))