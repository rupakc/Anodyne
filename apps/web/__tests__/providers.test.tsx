// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(cleanup);

import { ProvidersManager } from "@/app/app/providers/providers-manager";
import type { ProviderConfig, ProviderKind } from "@/lib/api";
import { baseMockApi } from "./mock-api";

const LLM_PROVIDER: ProviderConfig = {
  id: "prov-1",
  tenant_id: "tenant-1",
  name: "Local Ollama",
  provider: "ollama",
  model: "llama3.1",
  params: {},
  api_base: "http://localhost:11434",
  enabled: true,
};

const IMAGE_PROVIDER: ProviderConfig = {
  id: "prov-img-1",
  tenant_id: "tenant-1",
  name: "Stability",
  provider: "stability",
  model: "sdxl",
  params: {},
  api_base: null,
  enabled: false,
};

describe("ProvidersManager", () => {
  it("lists the LLM registry on first load", async () => {
    const listProviders = vi.fn().mockResolvedValue([LLM_PROVIDER]);
    const api = baseMockApi({ listProviders });

    render(<ProvidersManager api={api} />);

    expect(await screen.findByText("Local Ollama")).toBeInTheDocument();
    expect(screen.getByText("ollama/llama3.1")).toBeInTheDocument();
    expect(listProviders).toHaveBeenCalledWith("models");
  });

  it("switches registries when another modality tab is selected", async () => {
    const user = userEvent.setup();
    const listProviders = vi.fn().mockImplementation((kind: ProviderKind) =>
      Promise.resolve(kind === "image-providers" ? [IMAGE_PROVIDER] : [LLM_PROVIDER]),
    );
    const api = baseMockApi({ listProviders });

    render(<ProvidersManager api={api} />);
    await screen.findByText("Local Ollama");

    await user.click(screen.getByRole("tab", { name: "Image" }));

    expect(await screen.findByText("Stability")).toBeInTheDocument();
    expect(screen.getByText("stability/sdxl")).toBeInTheDocument();
    expect(listProviders).toHaveBeenCalledWith("image-providers");
    // The disabled provider is flagged as such.
    expect(screen.getByText("disabled")).toBeInTheDocument();
  });

  it("shows an empty state when a registry has no providers", async () => {
    const api = baseMockApi({ listProviders: vi.fn().mockResolvedValue([]) });

    render(<ProvidersManager api={api} />);

    expect(await screen.findByText(/no llm \/ text providers registered yet/i)).toBeInTheDocument();
  });

  it("registers a provider and refreshes the list", async () => {
    const user = userEvent.setup();
    const listProviders = vi
      .fn()
      .mockResolvedValueOnce([]) // initial
      .mockResolvedValue([LLM_PROVIDER]); // after register
    const registerProvider = vi.fn().mockResolvedValue(LLM_PROVIDER);
    const api = baseMockApi({ listProviders, registerProvider });

    render(<ProvidersManager api={api} />);
    await screen.findByText(/no llm \/ text providers registered yet/i);

    await user.type(screen.getByPlaceholderText("e.g. Local Ollama"), "Local Ollama");
    await user.type(screen.getByPlaceholderText("ollama"), "ollama");
    await user.type(screen.getByPlaceholderText("llama3.1 / sdxl / whisper-1"), "llama3.1");
    await user.click(screen.getByRole("button", { name: "Register provider" }));

    await waitFor(() =>
      expect(registerProvider).toHaveBeenCalledWith("models", {
        name: "Local Ollama",
        provider: "ollama",
        model: "llama3.1",
        api_key: undefined,
        api_base: undefined,
        params: undefined,
      }),
    );
    expect(await screen.findByText("Local Ollama")).toBeInTheDocument();
  });

  it("rejects invalid params JSON without calling the gateway", async () => {
    const user = userEvent.setup();
    const registerProvider = vi.fn();
    const api = baseMockApi({ listProviders: vi.fn().mockResolvedValue([]), registerProvider });

    render(<ProvidersManager api={api} />);
    await screen.findByText(/no llm \/ text providers/i);

    await user.type(screen.getByPlaceholderText("e.g. Local Ollama"), "X");
    await user.type(screen.getByPlaceholderText("ollama"), "ollama");
    await user.type(screen.getByPlaceholderText("llama3.1 / sdxl / whisper-1"), "m");
    await user.type(screen.getByPlaceholderText("{}"), "{{not json");
    await user.click(screen.getByRole("button", { name: "Register provider" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/valid json/i);
    expect(registerProvider).not.toHaveBeenCalled();
  });

  it("deletes a provider", async () => {
    const user = userEvent.setup();
    const deleteProvider = vi.fn().mockResolvedValue(undefined);
    const api = baseMockApi({
      listProviders: vi.fn().mockResolvedValue([LLM_PROVIDER]),
      deleteProvider,
    });

    render(<ProvidersManager api={api} />);
    const list = await screen.findByRole("list", { name: /providers/i });
    await user.click(within(list).getByRole("button", { name: /delete local ollama/i }));

    await waitFor(() => expect(deleteProvider).toHaveBeenCalledWith("models", "prov-1"));
    await waitFor(() => expect(screen.queryByText("Local Ollama")).not.toBeInTheDocument());
  });
});
