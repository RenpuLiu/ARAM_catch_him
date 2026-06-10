from __future__ import annotations

import argparse
from pathlib import Path

from llm_analysis import (
    DEFAULT_SKILL_PATH,
    LLMAnalysisError,
    build_analysis_payload,
    generate_analysis_report,
    list_analysis_skills,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an LLM ARAM analysis report.")
    parser.add_argument("--data", default="data", help="data directory")
    parser.add_argument(
        "--skill",
        action="append",
        default=None,
        help="SKILL.md path; pass multiple times to combine skills",
    )
    parser.add_argument("--min-partner-games", type=int, default=2)
    parser.add_argument("--recent-games", type=int, default=50)
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="override LLM_MAX_OUTPUT_TOKENS")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Responses API reasoning effort; use none/default to omit",
    )
    parser.add_argument("--dry-run", action="store_true", help="only build and print payload summary")
    parser.add_argument("--list-skills", action="store_true", help="list available analysis skills")
    args = parser.parse_args()

    if args.list_skills:
        for skill in list_analysis_skills():
            print(f"{skill.name}\t{skill.path}\t{skill.description}")
        return 0
    skill_paths = args.skill or [str(DEFAULT_SKILL_PATH)]

    try:
        if args.dry_run:
            payload = build_analysis_payload(
                data_dir=args.data,
                min_partner_games=args.min_partner_games,
                recent_games=args.recent_games,
            )
            print(
                {
                    "matches": payload["metadata"]["match_count"],
                    "participants": payload["metadata"]["participant_count"],
                    "frequent_allies": len(payload["frequent_allies"]),
                    "players_for_equal_analysis": len(payload["players_for_equal_analysis"]),
                    "skills": skill_paths,
                    "max_output_tokens": args.max_output_tokens,
                    "reasoning_effort": args.reasoning_effort,
                }
            )
            return 0

        result = generate_analysis_report(
            data_dir=args.data,
            skill_path=[Path(path) for path in skill_paths],
            min_partner_games=args.min_partner_games,
            recent_games=args.recent_games,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )
    except LLMAnalysisError as exc:
        print(f"LLM analysis error: {exc}")
        return 2

    print(result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
