# pgx pure-Go PostgreSQL driver; CGO gerekmez.
FROM golang:1.22-alpine AS builder
WORKDIR /build
RUN apk add --no-cache git
COPY go.mod ./
RUN go mod download || true
COPY . .
RUN go mod tidy
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /app/aggregator .

FROM alpine:3.19
RUN apk --no-cache add ca-certificates tzdata
COPY --from=builder /app/aggregator /app/aggregator
EXPOSE 8081
ENTRYPOINT ["/app/aggregator"]
