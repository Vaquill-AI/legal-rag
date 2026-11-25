"""Catalogue, naming and schema translation against a small inline OpenAPI document."""

from __future__ import annotations

import copy

import pytest

from legal_rag.mcp_bridge import (
    CatalogueError,
    RouteRef,
    SchemaTranslationError,
    ToolNameError,
    argument_name,
    build_catalogue,
    tool_name,
)

from .support import OPENAPI, ROUTES, SECTION_PATH, catalogue


def test_catalogue_has_one_tool_per_route_with_derived_names():
    specs = catalogue()
    assert set(specs) == {"statutes_get_section", "cases_search", "saved_cases_delete"}
    assert specs["statutes_get_section"].operation_id == "statutes.getSection"
    assert specs["cases_search"].path == "/us/cases/search"


def test_tool_names_are_injective_across_the_catalogue():
    names = [spec.name for spec in build_catalogue(openapi=OPENAPI, routes=ROUTES)]
    assert len(names) == len(set(names))


def test_two_operations_collapsing_onto_one_tool_name_refuse_to_build():
    # "a.bC" and "a.b.c" both become "a_b_c".
    assert tool_name("a.bC") == tool_name("a.b.c") == "a_b_c"
    document = {
        "paths": {
            "/one": {"get": {"operationId": "a.bC"}},
            "/two": {"get": {"operationId": "a.b.c"}},
        }
    }
    routes = [RouteRef("a.bC", "GET", "/one"), RouteRef("a.b.c", "GET", "/two")]
    with pytest.raises(ToolNameError, match="a_b_c"):
        build_catalogue(openapi=document, routes=routes)


@pytest.mark.parametrize("bad", ["", "bad name", "café.lookup"])
def test_tool_name_rejects_empty_and_illegal_operation_ids(bad):
    with pytest.raises(ToolNameError):
        tool_name(bad)


def test_input_schema_is_built_from_path_query_and_header_parameters():
    schema = catalogue()["statutes_get_section"].input_schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "code",
        "section",
        "effectiveDate",
        "includeHistory",
        "xRequestId",
    }
    assert schema["required"] == ["code", "section"]
    assert schema["properties"]["effectiveDate"] == {
        "type": "string",
        "format": "date",
        "description": "Return the text in force on this date.",
    }
    assert "$defs" not in schema


def test_input_schema_nests_the_body_and_relocates_only_reachable_refs():
    schema = catalogue()["cases_search"].input_schema
    assert set(schema["properties"]) == {"body", "court", "idempotencyKey"}
    assert schema["required"] == ["body"]
    assert schema["properties"]["body"] == {"$ref": "#/$defs/CaseSearchRequest"}
    assert set(schema["$defs"]) == {"CaseSearchRequest", "CourtLevel"}
    assert schema["$defs"]["CaseSearchRequest"]["properties"]["level"] == {
        "$ref": "#/$defs/CourtLevel"
    }
    assert "NeverReferenced" not in schema["$defs"]


def test_arguments_record_where_each_one_goes_on_the_wire():
    arguments = {a.name: (a.location, a.wire_name) for a in catalogue()["cases_search"].arguments}
    assert arguments == {
        "court": ("query", "court"),
        "idempotencyKey": ("header", "Idempotency-Key"),
        "body": ("body", "body"),
    }


def test_building_does_not_mutate_the_openapi_document():
    document = copy.deepcopy(OPENAPI)
    build_catalogue(openapi=document, routes=ROUTES)
    assert document == OPENAPI


def test_hints_follow_the_http_method():
    specs = catalogue()
    assert (specs["statutes_get_section"].read_only, specs["statutes_get_section"].idempotent) == (
        True,
        True,
    )
    assert (specs["cases_search"].read_only, specs["cases_search"].idempotent) == (False, False)
    assert specs["saved_cases_delete"].destructive is True
    assert specs["saved_cases_delete"].idempotent is True
    assert specs["statutes_get_section"].destructive is False


def test_description_carries_handler_prose_scopes_and_the_call_line():
    spec = catalogue()["cases_search"]
    assert spec.title == "Search case law"
    assert spec.scopes == ("cases:read", "cases:search")
    assert "Requires the scope cases:read, cases:search." in spec.description
    assert spec.description.endswith("Calls POST /us/cases/search (operation cases.search).")
    section = catalogue()["statutes_get_section"]
    assert section.description.startswith("Returns one section of a code")


def test_a_declared_operation_missing_from_the_document_is_refused():
    routes = [*ROUTES, RouteRef("statutes.listCodes", "GET", "/us/statutes")]
    with pytest.raises(CatalogueError, match="declared but absent"):
        build_catalogue(openapi=OPENAPI, routes=routes)


def test_a_documented_operation_nobody_declared_is_refused():
    with pytest.raises(CatalogueError, match="documented but not declared"):
        build_catalogue(openapi=OPENAPI, routes=ROUTES[:2])


def test_a_method_or_path_disagreement_is_refused():
    routes = [RouteRef("statutes.getSection", "POST", SECTION_PATH), *ROUTES[1:]]
    with pytest.raises(CatalogueError, match="declared POST"):
        build_catalogue(openapi=OPENAPI, routes=routes)


def test_required_arguments_promote_optional_ones_and_unknown_ones_are_refused():
    promoted = [
        RouteRef("statutes.getSection", "GET", SECTION_PATH, required_arguments=("effectiveDate",)),
        *ROUTES[1:],
    ]
    spec = {s.name: s for s in build_catalogue(openapi=OPENAPI, routes=promoted)}
    assert spec["statutes_get_section"].input_schema["required"] == [
        "code",
        "effectiveDate",
        "section",
    ]
    unknown = [
        RouteRef("statutes.getSection", "GET", SECTION_PATH, required_arguments=("nope",)),
        *ROUTES[1:],
    ]
    with pytest.raises(SchemaTranslationError, match="nope"):
        build_catalogue(openapi=OPENAPI, routes=unknown)


def test_a_reserved_header_parameter_is_refused():
    document = copy.deepcopy(OPENAPI)
    document["paths"]["/us/cases/search"]["post"]["parameters"].append(
        {"name": "Authorization", "in": "header", "schema": {"type": "string"}}
    )
    with pytest.raises(SchemaTranslationError, match="Authorization"):
        build_catalogue(openapi=document, routes=ROUTES)


def test_a_non_json_body_and_a_dangling_ref_are_refused():
    upload = copy.deepcopy(OPENAPI)
    upload["paths"]["/us/cases/search"]["post"]["requestBody"]["content"] = {
        "multipart/form-data": {"schema": {"type": "object"}}
    }
    with pytest.raises(SchemaTranslationError, match="application/json"):
        build_catalogue(openapi=upload, routes=ROUTES)

    dangling = copy.deepcopy(OPENAPI)
    del dangling["components"]["schemas"]["CourtLevel"]
    with pytest.raises(SchemaTranslationError, match="CourtLevel"):
        build_catalogue(openapi=dangling, routes=ROUTES)


def test_a_duplicate_operation_id_in_the_document_is_refused():
    document = copy.deepcopy(OPENAPI)
    document["paths"]["/elsewhere"] = {"get": {"operationId": "cases.search"}}
    with pytest.raises(CatalogueError, match="appears twice"):
        build_catalogue(openapi=document, routes=ROUTES)


def test_argument_names_camel_case_headers_and_pass_other_locations_through():
    assert argument_name("Idempotency-Key", "header") == "idempotencyKey"
    assert argument_name("X-Request-Id", "header") == "xRequestId"
    assert argument_name("effectiveDate", "query") == "effectiveDate"
    assert argument_name("code", "path") == "code"
