"""Agents: the plan, validate, generate, refine and verify loop around a language model.

``legal_rag.agents.actors`` holds the model-backed actors and a model-free plan validator.
``legal_rag.agents.generation`` orchestrates them around one buffered answer.
Import from those modules directly; this package imports nothing at load time.
"""
