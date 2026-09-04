import { useEffect, useRef } from "react";
import Plotly, {
  type PlotlyClickPoint,
  type PlotlyHTMLElement,
} from "plotly.js-dist-min";
import type { FigureSpec } from "../types";

const DEFAULT_HEIGHT = 380;
const DEFAULT_WIDTH = 640;

interface PlotlyFigureProps {
  figure: FigureSpec | null;
  onSelectPosition?: (position: number) => void;
  className?: string;
  "data-testid"?: string;
}

function positionFromPoint(point: PlotlyClickPoint): number | null {
  const custom = point.customdata;
  const value = Array.isArray(custom) ? custom[0] : custom;
  if (value !== undefined && value !== null && Number.isFinite(Number(value))) {
    return Number(value);
  }
  if (point.pointIndex !== undefined && Number.isFinite(Number(point.pointIndex))) {
    return Number(point.pointIndex);
  }
  return null;
}

export function PlotlyFigure({
  figure,
  onSelectPosition,
  className,
  "data-testid": testId,
}: PlotlyFigureProps) {
  const hostRef = useRef<PlotlyHTMLElement | null>(null);
  const selectRef = useRef(onSelectPosition);
  selectRef.current = onSelectPosition;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    if (!figure) {
      Plotly.purge(host);
      return;
    }

    const width = host.clientWidth || DEFAULT_WIDTH;
    const height = host.clientHeight || DEFAULT_HEIGHT;
    const layout = { ...figure.layout, width, height };
    const config = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    };

    Plotly.react(host, figure.data, layout, config).catch((reason: unknown) => {
      // A transient render failure should not take down the app, but it must
      // not stay invisible either.
      console.warn("Plotly render failed", reason);
    });
    host.removeAllListeners("plotly_click");
    host.on("plotly_click", (event) => {
      const first = event.points[0];
      if (!first) return;
      const position = positionFromPoint(first);
      if (position !== null) selectRef.current?.(position);
    });
  }, [figure]);

  useEffect(() => {
    const host = hostRef.current;
    return () => {
      if (host) Plotly.purge(host);
    };
  }, []);

  return (
    <div
      ref={(element) => {
        hostRef.current = element as PlotlyHTMLElement | null;
      }}
      className={className}
      data-testid={testId}
    />
  );
}
