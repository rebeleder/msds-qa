from src.prompt import (
    EntityContinueExtraction,
    EntityExtractionPrompt,
    EntityIfLoopExtraction,
    KeywordsExtraction,
)


class Prompt:
    @staticmethod
    def get_prompt(desc: str) -> str:
        return {
            "entity_extraction": EntityExtractionPrompt,
            "entity_continue_extraction": EntityContinueExtraction,
            "entity_if_loop_extraction": EntityIfLoopExtraction,
            "keywords_extraction": KeywordsExtraction,
            # "graph_field_sep": Prompt.get_graph_field_sep(),
        }[desc]

    @staticmethod
    def get_graph_field_sep() -> str:
        return "<SEP>"

    @staticmethod
    def get_default_tuple_delimiter() -> str:
        return "<|>"

    @staticmethod
    def get_default_record_delimiter() -> str:
        return "##"

    @staticmethod
    def get_default_completion_delimiter() -> str:
        return "<|COMPLETE|>"

    @staticmethod
    def get_process_tickers() -> list[str]:
        return ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    @staticmethod
    def get_default_entity_types() -> list[str]:
        return ["organization", "person", "location", "event"]
