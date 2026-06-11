import { Component, type ReactNode, type ErrorInfo } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(_error: Error, _info: ErrorInfo) {}

  render() {
    if (!this.state.hasError) return this.props.children

    if (this.props.fallback) return this.props.fallback

    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-6">
        <div className="text-[#ff1744] text-4xl">⚠</div>
        <div>
          <p className="text-sm font-semibold text-[#e8e8e8]">This page crashed</p>
          <p className="text-xs text-[#555] mt-1 font-mono max-w-md">
            {this.state.error?.message ?? 'Unknown error'}
          </p>
        </div>
        <button
          onClick={() => this.setState({ hasError: false, error: null })}
          className="px-4 py-2 text-xs bg-[#1e1e1e] hover:bg-[#2a2a2a] border border-[#2a2a2a] rounded text-[#888] transition-colors"
        >
          Try again
        </button>
      </div>
    )
  }
}
