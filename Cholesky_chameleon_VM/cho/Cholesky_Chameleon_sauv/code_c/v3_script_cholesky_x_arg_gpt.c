#include <chameleon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

/* ------------------------------------------------------------------
 * cholesky_namedargs.c  (ALL ARGS REQUIRED VERSION)
 *
 * Tous les parametres doivent etre explicitement passes en options
 * nommees. AUCUNE VALEUR PAR DEFAUT n'est assumee. Si une option est
 * absente, le programme affiche l'usage et termine en erreur.
 *
 * Cela t'evite les erreurs de position (+ lisibilite) en gardant la
 * flexibilite de ton ancienne version a 20 arguments.
 *
 * Les commentaires d'origine fournis par l'utilisateur sont preserves
 * (encapsules dans des blocs /* ... * / pour compiler proprement).
 * ------------------------------------------------------------------ */

/* ---------------- Mapping helpers (error if unknown) ---------------- */
static cham_flttype_t map_dtyp_from_string(const char *s, int *ok)
{
    if (!s || !*s) { *ok = 0; return ChamRealDouble; }
    if (!strcmp(s,"d") || !strcmp(s,"D") || !strcmp(s,"0")) { *ok=1; return ChamRealDouble; }
    if (!strcmp(s,"s") || !strcmp(s,"S") || !strcmp(s,"1")) { *ok=1; return ChamRealFloat; }
    if (!strcmp(s,"z") || !strcmp(s,"Z") || !strcmp(s,"2")) { *ok=1; return ChamComplexDouble; }
    if (!strcmp(s,"c") || !strcmp(s,"C") || !strcmp(s,"3")) { *ok=1; return ChamComplexFloat; }
    *ok = 0; return ChamRealDouble;
}

static cham_uplo_t map_uplo_from_string(const char *s, int *ok)
{
    if (!s || !*s) { *ok = 0; return ChamLower; }
    if (!strcmp(s,"L") || !strcmp(s,"l") || !strcmp(s,"0")) { *ok=1; return ChamLower; }
    if (!strcmp(s,"U") || !strcmp(s,"u") || !strcmp(s,"1")) { *ok=1; return ChamUpper; }
    if (!strcmp(s,"B") || !strcmp(s,"b") || !strcmp(s,"2")) { *ok=1; return ChamUpperLower; }
    *ok = 0; return ChamLower;
}

/* safe converters */
static long              s2l   (const char *s){ return strtol (s, NULL, 10); }
static unsigned long     s2ul  (const char *s){ return strtoul (s, NULL, 10); }
static long long         s2ll  (const char *s){ return strtoll (s, NULL, 10); }
static unsigned long long s2ull(const char *s){ return strtoull(s, NULL, 10); }
static double            s2dbl (const char *s){ return strtod (s, NULL); }

/* ------------------------------------------------------------------ */
static void usage(const char *prog)
{
    fprintf(stderr,
"Usage: %s --N INT --NB INT --ncpu INT --ngpu INT --mat none|user --dtyp d|s|z|c \\\n\n          --mb INT --nb INT --bsiz INT --lm INT --ln INT --i INT --j INT \\\n\n          --m INT --n INT --p INT --q INT --bump DOUBLE --uplo L|U|B --seed ULL\n\n"
"ALL options are required. No defaults.\n\n"
"Example:\n  %s --N 3000 --NB 256 --ncpu 4 --ngpu 1 --mat none --dtyp d \\\n\n     --mb 256 --nb 256 --bsiz 65536 --lm 3000 --ln 3000 --i 0 --j 0 \\\n\n     --m 3000 --n 3000 --p 1 --q 1 --bump 3000 --uplo L --seed 51\n",
            prog, prog);
}

int main(int argc, char **argv)
{
    /* placeholders */
    const char *N_s=NULL,*NB_s=NULL,*ncpu_s=NULL,*ngpu_s=NULL,*mat_s=NULL,*dtyp_s=NULL;
    const char *mb_s=NULL,*nb_s=NULL,*bsiz_s=NULL,*lm_s=NULL,*ln_s=NULL,*i_s=NULL,*j_s=NULL;
    const char *m_s=NULL,*n_s=NULL,*p_s=NULL,*q_s=NULL,*bump_s=NULL,*uplo_s=NULL,*seed_s=NULL;

    int opt, lidx;
    static struct option longopts[] = {
        {"N",     required_argument, 0, 'N'},
        {"NB",    required_argument, 0, 'B'},
        {"ncpu",  required_argument, 0,  0 },
        {"ngpu",  required_argument, 0,  0 },
        {"mat",   required_argument, 0,  0 },
        {"dtyp",  required_argument, 0,  0 },
        {"mb",    required_argument, 0,  0 },
        {"nb",    required_argument, 0,  0 },
        {"bsiz",  required_argument, 0,  0 },
        {"lm",    required_argument, 0,  0 },
        {"ln",    required_argument, 0,  0 },
        {"i",     required_argument, 0,  0 },
        {"j",     required_argument, 0,  0 },
        {"m",     required_argument, 0,  0 },
        {"n",     required_argument, 0,  0 },
        {"p",     required_argument, 0,  0 },
        {"q",     required_argument, 0,  0 },
        {"bump",  required_argument, 0,  0 },
        {"uplo",  required_argument, 0,  0 },
        {"seed",  required_argument, 0,  0 },
        {"help",  no_argument,       0, 'h'},
        {0,0,0,0}
    };

    while ((opt = getopt_long(argc, argv, "h", longopts, &lidx)) != -1) {
        switch(opt) {
        case 'h': usage(argv[0]); return EXIT_SUCCESS;
        case 'N': N_s  = optarg; break;
        case 'B': NB_s = optarg; break;
        case 0: {
            const char *name = longopts[lidx].name;
            if      (!strcmp(name,"ncpu")) ncpu_s = optarg;
            else if (!strcmp(name,"ngpu")) ngpu_s = optarg;
            else if (!strcmp(name,"mat"))  mat_s  = optarg;
            else if (!strcmp(name,"dtyp")) dtyp_s = optarg;
            else if (!strcmp(name,"mb"))   mb_s   = optarg;
            else if (!strcmp(name,"nb"))   nb_s   = optarg;
            else if (!strcmp(name,"bsiz")) bsiz_s = optarg;
            else if (!strcmp(name,"lm"))   lm_s   = optarg;
            else if (!strcmp(name,"ln"))   ln_s   = optarg;
            else if (!strcmp(name,"i"))    i_s    = optarg;
            else if (!strcmp(name,"j"))    j_s    = optarg;
            else if (!strcmp(name,"m"))    m_s    = optarg;
            else if (!strcmp(name,"n"))    n_s    = optarg;
            else if (!strcmp(name,"p"))    p_s    = optarg;
            else if (!strcmp(name,"q"))    q_s    = optarg;
            else if (!strcmp(name,"bump")) bump_s = optarg;
            else if (!strcmp(name,"uplo")) uplo_s = optarg;
            else if (!strcmp(name,"seed")) seed_s = optarg;
            break; }
        default: usage(argv[0]); return EXIT_FAILURE;
        }
    }

    /* Check ALL required */
    if (!N_s||!NB_s||!ncpu_s||!ngpu_s||!mat_s||!dtyp_s||!mb_s||!nb_s||!bsiz_s||!lm_s||!ln_s||!i_s||!j_s||!m_s||!n_s||!p_s||!q_s||!bump_s||!uplo_s||!seed_s) {
        fprintf(stderr,"Error: all options are required. Missing at least one.\n");
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    /* Parse numeric */
    int NCPU = (int)s2l(ncpu_s);
    int NGPU = (int)s2l(ngpu_s);
    int N    = (int)s2l(N_s);
    int NB   = (int)s2l(NB_s);
    int mb   = (int)s2l(mb_s);
    int nb   = (int)s2l(nb_s);
    long bsiz_l = s2l(bsiz_s);
    int lm   = (int)s2l(lm_s);
    int ln   = (int)s2l(ln_s);
    int ioff = (int)s2l(i_s);
    int joff = (int)s2l(j_s);
    int m    = (int)s2l(m_s);
    int n    = (int)s2l(n_s);
    int p    = (int)s2l(p_s);
    int q    = (int)s2l(q_s);
    double bump = s2dbl(bump_s);
    unsigned long long seed = s2ull(seed_s);

    /* map type+uplo */
    int dtyp_ok=0,uplo_ok=0;
    cham_flttype_t dtyp = map_dtyp_from_string(dtyp_s,&dtyp_ok);
    cham_uplo_t    uplo = map_uplo_from_string(uplo_s,&uplo_ok);
    if (!dtyp_ok) { fprintf(stderr,"Error: invalid --dtyp %s\n", dtyp_s); return EXIT_FAILURE; }
    if (!uplo_ok) { fprintf(stderr,"Error: invalid --uplo %s\n", uplo_s); return EXIT_FAILURE; }

    /* interpret mat */
    int mat_user = (strcmp(mat_s,"none")&&strcmp(mat_s,"NULL")&&strcmp(mat_s,"0"));
    void *mat = NULL;
    if (mat_user) {
        size_t elems = (size_t)lm * (size_t)ln;
        size_t bytes = elems;
        switch(dtyp){
        case ChamRealDouble:    bytes *= sizeof(double); break;
        case ChamRealFloat:     bytes *= sizeof(float); break;
        case ChamComplexDouble: bytes *= 2*sizeof(double); break;
        case ChamComplexFloat:  bytes *= 2*sizeof(float); break;
        default:                bytes *= sizeof(double); break;
        }
        mat = malloc(bytes);
        if (!mat) {
            fprintf(stderr,"Failed to malloc user matrix (%zu bytes)\n", bytes);
            return EXIT_FAILURE;
        }
    }

    /* Validate dims strictly (since no defaults) */
    if (N<=0||NB<=0||mb<=0||nb<=0||lm<=0||ln<=0||m<=0||n<=0||p<=0||q<=0) {
        fprintf(stderr,"Error: dimension arguments must be >0.\n");
        return EXIT_FAILURE;
    }
    if (bsiz_l < (long)mb*(long)nb) {
        fprintf(stderr,"Error: --bsiz < mb*nb (bsiz=%ld mb=%d nb=%d).\n",bsiz_l,mb,nb);
        return EXIT_FAILURE;
    }
    if (ioff < 0 || joff < 0 || ioff >= lm || joff >= ln) {
        fprintf(stderr,"Error: invalid offsets i=%d j=%d (lm=%d ln=%d).\n",ioff,joff,lm,ln);
        return EXIT_FAILURE;
    }
    if (ioff+m > lm || joff+n > ln) {
        fprintf(stderr,"Error: submatrix (i=%d,m=%d) outside lm=%d OR (j=%d,n=%d) outside ln=%d.\n",ioff,m,lm,joff,n,ln);
        return EXIT_FAILURE;
    }
    if (bump == 0.0) {
        fprintf(stderr,"Warning: bump==0 -> matrix may not be SPD.\n");
    }

    size_t bsiz = (size_t)bsiz_l;

    int info;
    CHAM_desc_t *descA;

    /* ------------------- ORIGINAL USER COMMENTS (kept) ------------------- */
    /*
    But: initialize the Chameleon runtime (context, StarPU backend, optional GPU stack, env parsing); must be called before creating descriptors or invoking kernels.
    */
    CHAMELEON_Init(NCPU,NGPU);

    /*
    But : creer un descripteur de matrice tuilee que chameleon utilise pour raisonner sur la distribution memoire, la taille des tuiles et sur les sous matrices.
    */
    CHAMELEON_Desc_Create(&descA, mat, dtyp, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);

    /*
    But : remplir le descripteur A avec une matrice aleatoire symetrique ; devient SPD si le bump est assez grand.
    */
    CHAMELEON_dplgsy_Tile(bump, uplo, descA, seed);

    /* chronometrage */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* CHAMELEON_dpotrf_Tile(uplo,A)
     * Purpose                  : in-place Cholesky factorization A = L*L^T or U^T*U of SPD matrix stored in tiles.
     */
    info = CHAMELEON_dpotrf_Tile(uplo, descA);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    /* flop count: use m (height submatrix) assume square m==n */
    double dim = (double)((m<n)?m:n);
    double gflops = (1.0/3.0) * dim * dim * dim / (time_sec * 1e9);

    printf("N=%d NB=%d ncpu=%d ngpu=%d p=%d q=%d bump=%g uplo=%d seed=%llu\n",
           N, NB, NCPU, NGPU, p, q, bump, (int)uplo, (unsigned long long)seed);
    printf("Time: %.6f s\n", time_sec);
    printf("Performance: %.2f Gflop/s\n", gflops);

    if (info != 0) {
        fprintf(stderr, "Erreur dans CHAMELEON_dpotrf_Tile: %d\n", info);
    }

    CHAMELEON_Desc_Destroy(&descA);
    CHAMELEON_Finalize();
    if (mat) free(mat);
    return (info != 0);
}
