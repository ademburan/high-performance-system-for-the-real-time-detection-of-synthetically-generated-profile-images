// Package main - GANEye Traffic Generator.
//
// Kullanım:
//
//	go run . -src=TwitterGAN_profiles.tar.gz -target=http://localhost:8080/detect -rps=50 -duration=60s
//
// -src hem bir .tar.gz arşivi hem de düz bir klasör olabilir. Arşiv verildiğinde
// içindeki tüm .jpg/.jpeg/.png dosyaları belleğe okunur (küçük profil görselleri
// için uygundur). Daha sonra rate-limited goroutine havuzu üzerinden
// multipart/form-data POST olarak gönderilir.
//
// Çıktı: canlı metrik log'u + sonunda özet (gönderilen, başarılı, hata, p50/p95 latency).
package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"mime/multipart"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/time/rate"
)

type sample struct {
	name string
	data []byte
}

var imageExts = map[string]struct{}{
	".jpg":  {},
	".jpeg": {},
	".png":  {},
	".webp": {},
}

func isImage(name string) bool {
	_, ok := imageExts[strings.ToLower(filepath.Ext(name))]
	return ok
}

// loadFromTarGz - tar.gz arşivini belleğe açar. Büyük arşivler için
// streaming alternatifi gerekebilir; 100k dosyaya kadar bu yaklaşım iyidir.
func loadFromTarGz(path string) ([]sample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("gzip: %w", err)
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	var out []sample
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("tar next: %w", err)
		}
		if hdr.Typeflag != tar.TypeReg || !isImage(hdr.Name) {
			continue
		}
		buf := make([]byte, hdr.Size)
		if _, err := io.ReadFull(tr, buf); err != nil {
			return nil, fmt.Errorf("tar read %s: %w", hdr.Name, err)
		}
		out = append(out, sample{name: filepath.Base(hdr.Name), data: buf})
	}
	return out, nil
}

// loadFromDir - verilen klasörü recursive tarayıp görselleri okur.
func loadFromDir(path string) ([]sample, error) {
	var out []sample
	err := filepath.Walk(path, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() || !isImage(info.Name()) {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		out = append(out, sample{name: info.Name(), data: data})
		return nil
	})
	return out, err
}

// buildMultipart - görseli API'nin beklediği multipart form'a sarar.
func buildMultipart(s sample) (*bytes.Buffer, string, error) {
	body := &bytes.Buffer{}
	w := multipart.NewWriter(body)
	part, err := w.CreateFormFile("image", s.name)
	if err != nil {
		return nil, "", err
	}
	if _, err := part.Write(s.data); err != nil {
		return nil, "", err
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return body, w.FormDataContentType(), nil
}

// resultLine - loadgen'in opsiyonel CSV çıktısı için bir satır.
type resultLine struct {
	Filename string
	UUID     string
	Status   int
}

func main() {
	src := flag.String("src", "", "kaynak tar.gz dosyası VEYA klasör yolu")
	target := flag.String("target", "http://localhost:8080/detect", "Gateway endpoint")
	rps := flag.Float64("rps", 50, "saniyede istek sayısı")
	duration := flag.Duration("duration", 0, "test süresi; 0 = tüm görseller tek tur")
	workers := flag.Int("workers", 32, "paralel worker sayısı")
	mapOut := flag.String("mapout", "", "filename->uuid eşleşmesi için CSV çıktı dosyası (opsiyonel)")
	seed := flag.Int64("seed", 42, "random seed")
	flag.Parse()

	if *src == "" {
		log.Fatal("-src gerekli (tar.gz dosyası veya klasör)")
	}

	log.Printf("[loadgen] loading source=%s", *src)
	var samples []sample
	var err error
	fi, err := os.Stat(*src)
	if err != nil {
		log.Fatalf("stat: %v", err)
	}
	if fi.IsDir() {
		samples, err = loadFromDir(*src)
	} else {
		samples, err = loadFromTarGz(*src)
	}
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	if len(samples) == 0 {
		log.Fatal("hiç görsel bulunamadı")
	}
	log.Printf("[loadgen] loaded %d images", len(samples))

	// DÜZELTME: v1'de rng oluşturulup kullanılmıyordu; görseller hiç
	// karıştırılmıyordu. Artık seed ile deterministik shuffle yapılıyor.
	rng := rand.New(rand.NewSource(*seed))
	rng.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })

	// HTTP client - yüksek concurrency için tuned.
	client := &http.Client{
		Timeout: 15 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        1000,
			MaxIdleConnsPerHost: 500,
			MaxConnsPerHost:     500,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	limiter := rate.NewLimiter(rate.Limit(*rps), int(*rps))

	ctx, cancel := context.WithCancel(context.Background())
	if *duration > 0 {
		ctx, cancel = context.WithTimeout(ctx, *duration)
	}
	defer cancel()

	// SIGINT/TERM ile temiz kapanış.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() { <-sig; log.Println("[loadgen] interrupt"); cancel() }()

	var sent, ok, failed int64
	latencies := make([]time.Duration, 0, 10000)
	var latMu sync.Mutex
	var resultsMu sync.Mutex
	resultsOut := make([]resultLine, 0, len(samples))

	jobs := make(chan sample, *workers*2)
	wg := &sync.WaitGroup{}
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for s := range jobs {
				body, ct, err := buildMultipart(s)
				if err != nil {
					atomic.AddInt64(&failed, 1)
					continue
				}
				req, err := http.NewRequestWithContext(ctx, http.MethodPost, *target, body)
				if err != nil {
					atomic.AddInt64(&failed, 1)
					continue
				}
				req.Header.Set("Content-Type", ct)

				t0 := time.Now()
				resp, err := client.Do(req)
				d := time.Since(t0)

				atomic.AddInt64(&sent, 1)
				if err != nil {
					atomic.AddInt64(&failed, 1)
					continue
				}

				latMu.Lock()
				latencies = append(latencies, d)
				latMu.Unlock()

				// UUID'yi mapping csv'sine yazmak için response'u parse et.
				var uuidStr string
				if *mapOut != "" && resp.StatusCode == http.StatusAccepted {
					var r struct {
						UUID string `json:"uuid"`
					}
					_ = json.NewDecoder(resp.Body).Decode(&r)
					uuidStr = r.UUID
				}
				// Body'yi tamamen drain et (conn reuse için).
				_, _ = io.Copy(io.Discard, resp.Body)
				resp.Body.Close()

				if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					atomic.AddInt64(&ok, 1)
				} else {
					atomic.AddInt64(&failed, 1)
				}

				if *mapOut != "" {
					resultsMu.Lock()
					resultsOut = append(resultsOut, resultLine{
						Filename: s.name,
						UUID:     uuidStr,
						Status:   resp.StatusCode,
					})
					resultsMu.Unlock()
				}
			}
		}()
	}

	// İstatistik ticker.
	stop := make(chan struct{})
	go func() {
		tk := time.NewTicker(2 * time.Second)
		defer tk.Stop()
		var lastSent int64
		var lastT = time.Now()
		for {
			select {
			case <-stop:
				return
			case <-tk.C:
				now := time.Now()
				s := atomic.LoadInt64(&sent)
				o := atomic.LoadInt64(&ok)
				f := atomic.LoadInt64(&failed)
				elapsed := now.Sub(lastT).Seconds()
				cur := float64(s-lastSent) / elapsed
				log.Printf("[loadgen] sent=%d ok=%d fail=%d current_rps=%.1f", s, o, f, cur)
				lastSent, lastT = s, now
			}
		}
	}()

	// Ana gönderme döngüsü: duration=0 ise tüm görselleri bir kez gönder.
	start := time.Now()
	i := 0
	for {
		if ctx.Err() != nil {
			break
		}
		if *duration == 0 && i >= len(samples) {
			break
		}
		if err := limiter.Wait(ctx); err != nil {
			break
		}
		s := samples[i%len(samples)]
		select {
		case jobs <- s:
		case <-ctx.Done():
		}
		i++
	}
	close(jobs)
	wg.Wait()
	close(stop)

	total := time.Since(start)
	latMu.Lock()
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	var p50, p95, p99 time.Duration
	if n := len(latencies); n > 0 {
		p50 = latencies[n*50/100]
		p95 = latencies[n*95/100]
		p99 = latencies[min(n-1, n*99/100)]
	}
	latMu.Unlock()

	log.Printf("[loadgen] DONE total=%s sent=%d ok=%d fail=%d avg_rps=%.1f p50=%s p95=%s p99=%s",
		total, sent, ok, failed, float64(sent)/total.Seconds(), p50, p95, p99)

	// Mapping CSV'sini dump et - README'deki doğruluk testi için gerekli.
	if *mapOut != "" {
		f, err := os.Create(*mapOut)
		if err != nil {
			log.Printf("mapout create: %v", err)
			return
		}
		defer f.Close()
		w := csv.NewWriter(f)
		_ = w.Write([]string{"filename", "uuid", "http_status"})
		for _, r := range resultsOut {
			_ = w.Write([]string{r.Filename, r.UUID, fmt.Sprintf("%d", r.Status)})
		}
		w.Flush()
		log.Printf("[loadgen] mapping written to %s (%d rows)", *mapOut, len(resultsOut))
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
