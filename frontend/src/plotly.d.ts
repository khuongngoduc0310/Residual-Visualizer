declare module "plotly.js-dist-min" {
  export interface PlotlyClickPoint {
    customdata?: number | number[];
    pointIndex?: number;
  }

  export interface PlotlyClickEvent {
    points: PlotlyClickPoint[];
  }

  export interface PlotlyHTMLElement extends HTMLElement {
    on(event: "plotly_click", callback: (event: PlotlyClickEvent) => void): void;
    removeAllListeners(event?: string): void;
  }

  interface PlotlyStatic {
    newPlot(
      element: PlotlyHTMLElement,
      data: unknown[],
      layout: Record<string, unknown>,
      config?: Record<string, unknown>,
    ): Promise<unknown>;
    react(
      element: PlotlyHTMLElement,
      data: unknown[],
      layout: Record<string, unknown>,
      config?: Record<string, unknown>,
    ): Promise<unknown>;
    purge(element: PlotlyHTMLElement): void;
  }

  const Plotly: PlotlyStatic;
  export default Plotly;
}
